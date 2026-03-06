#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <mma.h>
#include <optional>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <vector>


// Simple function to check if two bf16 values are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====
using namespace nvcuda;

// WMMA tile sizes
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
// Block tile sizes for Tensor-Core kernel
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;

// Fallback (scalar/shared memory) kernel, copied and renamed from original
__global__ void matrix_mul_bf16_fallback_kernel_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    const int Mtile = 64;
    const int Ntile = 64;
    const int Ktile = 16;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int rowBase = blockRow * Mtile;
    int colBase = blockCol * Ntile;
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + Mtile * Ktile;
    enum { Tr = Mtile / 16, Tc = Ntile / 16 };
    float regC[Tr][Tc] = {0.0f};
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    for (int offsetK = 0; offsetK < K; offsetK += Ktile) {
        int currentK = min(Ktile, K - offsetK);
        int totalA = Mtile * currentK;
        for (int idx = tid; idx < totalA; idx += blockSize) {
            int i = idx / currentK;
            int k = idx % currentK;
            int globalRow = rowBase + i;
            int globalK   = offsetK + k;
            float aval = 0.0f;
            if (globalRow < M && globalK < K) aval = __bfloat162float(A[globalRow * K + globalK]);
            As[i * Ktile + k] = aval;
        }
        int totalB = currentK * Ntile;
        for (int idx = tid; idx < totalB; idx += blockSize) {
            int k = idx / Ntile;
            int j = idx % Ntile;
            int globalK = offsetK + k;
            int globalCol = colBase + j;
            float bval = 0.0f;
            if (globalK < K && globalCol < N) bval = __bfloat162float(B[globalK * N + globalCol]);
            Bs[k * Ntile + j] = bval;
        }
        __syncthreads();
        for (int kk = 0; kk < currentK; ++kk) {
            float aVals[Tr];
            for (int i = 0; i < Tr; ++i) {
                int localRow = threadIdx.y * Tr + i;
                aVals[i] = As[localRow * Ktile + kk];
            }
            for (int j = 0; j < Tc; ++j) {
                int localCol = threadIdx.x * Tc + j;
                float bval = Bs[kk * Ntile + localCol];
                for (int i = 0; i < Tr; ++i) regC[i][j] += aVals[i] * bval;
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < Tr; ++i) {
        int globalRow = rowBase + threadIdx.y * Tr + i;
        if (globalRow >= M) continue;
        for (int j = 0; j < Tc; ++j) {
            int globalCol = colBase + threadIdx.x * Tc + j;
            if (globalCol >= N) continue;
            float cval = regC[i][j];
            C[globalRow * N + globalCol] = __float2bfloat16(cval);
        }
    }
}

// Tensor-Core accelerated kernel
__global__ void matrix_mul_bf16_wmma_kernel_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    // Block tile indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int rowBase = blockRow * TILE_M;
    int colBase = blockCol * TILE_N;

    extern __shared__ __nv_bfloat16 shmem[];
    __nv_bfloat16* shA = shmem;
    __nv_bfloat16* shB = shmem + TILE_M * WMMA_K;

    // Thread and warp indexing
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int warpId = tid / 32;
    int laneId = tid % 32;
    int warpRow = warpId / (TILE_N / WMMA_N);
    int warpCol = warpId % (TILE_N / WMMA_N);

    // Accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    // Loop over K
    for (int offsetK = 0; offsetK < K; offsetK += WMMA_K) {
        int currentK = min(WMMA_K, K - offsetK);
        // Load A tile into shared memory
        int numElemA = TILE_M * currentK;
        for (int idx = tid; idx < numElemA; idx += blockSize) {
            int i = idx / currentK;
            int k = idx % currentK;
            int globalRow = rowBase + i;
            int globalK   = offsetK + k;
            __nv_bfloat16 aval = __float2bfloat16(0);
            if (globalRow < M && globalK < K) aval = A[globalRow * K + globalK];
            shA[i * WMMA_K + k] = aval;
        }
        // Load B tile into shared memory
        int numElemB = currentK * TILE_N;
        for (int idx = tid; idx < numElemB; idx += blockSize) {
            int k = idx / TILE_N;
            int j = idx % TILE_N;
            int globalK = offsetK + k;
            int globalCol = colBase + j;
            __nv_bfloat16 bval = __float2bfloat16(0);
            if (globalK < K && globalCol < N) bval = B[globalK * N + globalCol];
            shB[k * TILE_N + j] = bval;
        }
        __syncthreads();

        // Load fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> bFrag;
        
        const __nv_bfloat16* tileAPtr = shA + warpRow * WMMA_M * WMMA_K;
        const __nv_bfloat16* tileBPtr = shB + warpCol * WMMA_N;
        wmma::load_matrix_sync(aFrag, tileAPtr, WMMA_K);
        wmma::load_matrix_sync(bFrag, tileBPtr, TILE_N);
        
        // Perform the matrix multiplication
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        __syncthreads();
    }

    // Store the result back to C
    int outRow = rowBase + warpRow * WMMA_M;
    int outCol = colBase + warpCol * WMMA_N;
    if (outRow < M && outCol < N) {
        // Each fragment lane writes one element
        // Convert the fragment to shared memory temp and then write by lane
        float cTmp[WMMA_M * WMMA_N];
        wmma::store_matrix_sync(cTmp, cFrag, WMMA_N, wmma::mem_row_major);
        for (int i = 0; i < WMMA_M; ++i) {
            int targetRow = outRow + i;
            if (targetRow >= M) continue;
            for (int j = 0; j < WMMA_N; ++j) {
                int targetCol = outCol + j;
                if (targetCol >= N) continue;
                int idx = i * WMMA_N + j;
                C[targetRow * N + targetCol] = __float2bfloat16(cTmp[idx]);
            }
        }
    }
}

// External C wrapper: always use fallback
extern "C" void matrix_mul_bf16_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16*       C,
    int                  M,
    int                  N,
    int                  K) 
{
    // Always use the fallback (shared-memory + scalar) kernel:
    dim3 block(16, 16, 1);
    dim3 grid((N + 64 - 1) / 64,
              (M + 64 - 1) / 64,
              1);
    // shared bytes = floats for A-tile (64×16) + floats for B-tile (16×64)
    size_t shared_bytes = sizeof(float) * (64 * 16 + 16 * 64);

    matrix_mul_bf16_fallback_kernel_optimized
      <<<grid, block, shared_bytes>>>(A, B, C, M, N, K);
    // force sync so errors get caught immediately
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void matrix_mul_bf16_kernel(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    __nv_bfloat16* C, 
    int M, 
    int N, 
    int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = __bfloat162float(A[row * K + k]);
            float b = __bfloat162float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

extern "C" void matrix_mul_bf16_origin(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    __nv_bfloat16* C, 
    int M, 
    int N, 
    int K) {
    
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    matrix_mul_bf16_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int M;
    int N;
    int K;
    std::vector<__nv_bfloat16> A;
    std::vector<__nv_bfloat16> B;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> M_list = {1 << 6, 1 << 8, 1 << 10, 1 << 12, 1 << 14};
    int N = 2048;
    int K = 4096;

    // Use fixed seed for reproducibility
    std::random_device rd;
    std::mt19937 rng(rd());  // Random seed for testing
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);

    for (int i = 0; i < M_list.size(); i++) {
        TestCase test_case;
        test_case.M = M_list[i];  // Larger size for better timing
        test_case.N = N;
        test_case.K = K;
        
        int A_item = test_case.M * test_case.K;
        int B_item = test_case.K * test_case.N;

        test_case.A.resize(A_item);
        test_case.B.resize(B_item);
        
        for (int ii = 0; ii < A_item; ii++) {
            test_case.A[ii] = __float2bfloat16(input_dist(rng));
        }
        for (int ii = 0; ii < B_item; ii++) {
            test_case.B[ii] = __float2bfloat16(input_dist(rng));
        }
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: M: %d, K: %d, N: %d. Complexity: %ld\n", test_case.M, test_case.K, test_case.N, (long)test_case.M * test_case.K * test_case.N);
}

// Function to warm up GPU and stabilize frequency
void stabilize_gpu() {
    // Create a dummy kernel to warm up GPU
    float *d_temp;
    cudaMalloc(&d_temp, sizeof(float));
    cudaFree(d_temp);
    
    // Small delay to let GPU stabilize
    for (volatile int i = 0; i < 10000; i++); // Busy wait
}

// Function to measure kernel performance with multiple iterations
template<typename KernelFunc>
float measure_kernel_performance(KernelFunc kernel, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        kernel();
    }
    
    // Measure multiple iterations
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return total_time ;  // Average time per iteration
}

// Function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    /* ================  Prepare data  ================ */
    std::vector<TestCase> test_case_list;
    load_test_case(test_case_list);

    for (const auto& test_case : test_case_list) {
        // Calculate sizes
        const int A_item = test_case.M * test_case.K;
        const int B_item = test_case.K * test_case.N;
        const int C_item = test_case.M * test_case.N;
        size_t A_size = A_item * sizeof(__nv_bfloat16);
        size_t B_size = B_item * sizeof(__nv_bfloat16);
        size_t C_size = C_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_A = (__nv_bfloat16*)malloc(A_size);
        __nv_bfloat16* h_B = (__nv_bfloat16*)malloc(B_size);
        __nv_bfloat16* h_C = (__nv_bfloat16*)malloc(C_size);
        __nv_bfloat16* h_C_optimized = (__nv_bfloat16*)malloc(C_size);

        if (!h_A || !h_B || !h_C || !h_C_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_A, test_case.A.data(), A_size);
        memcpy(h_B, test_case.B.data(), B_size);

        // GPU memory allocation
        __nv_bfloat16 *d_A, *d_B, *d_C, *d_C_optimized;

        checkCudaError(cudaMalloc((void**)&d_A, A_size), "Allocating d_A");
        checkCudaError(cudaMalloc((void**)&d_B, B_size), "Allocating d_B");
        checkCudaError(cudaMalloc((void**)&d_C, C_size), "Allocating d_C");
        checkCudaError(cudaMalloc((void**)&d_C_optimized, C_size), "Allocating d_C_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice), "Copying h_A to d_A");
        checkCudaError(cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice), "Copying h_B to d_B");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            matrix_mul_bf16_origin(d_A, d_B, d_C, test_case.M, test_case.N, test_case.K);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            matrix_mul_bf16_optimized(d_A, d_B, d_C_optimized, test_case.M, test_case.N, test_case.K);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost), "Copying d_C to h_C");
        checkCudaError(cudaMemcpy(h_C_optimized, d_C_optimized, C_size, cudaMemcpyDeviceToHost), "Copying d_C_optimized to h_C_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < C_item; i++) {
            if (!bfloat16_equals(h_C[i], h_C_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_C[i]), __bfloat162float(h_C_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_C_optimized);

        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_optimized);
    }

    return 0;
}