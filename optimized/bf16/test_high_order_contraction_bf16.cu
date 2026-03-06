#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <optional>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <vector>


// Simple function to check if two floats are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====
// Optimized configuration constants
static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 64;
static constexpr int BLOCK_K = 32;
static constexpr int THREADS_X = 16;
static constexpr int THREADS_Y = 16;
static constexpr int REG_M = BLOCK_M / THREADS_Y;  // 4
static constexpr int REG_N = BLOCK_N / THREADS_X;  // 4

// Optimized kernel with shared-memory tiling, register blocking, and bf16->fp32 computation
__global__ void high_order_contraction_bf16_kernel_optimized(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim)
{
    // Compute flattened dimensions
    int M = a_dim * b_dim;
    int K = x_dim * y_dim;
    int N = c_dim;

    // Block indices
    int block_m = blockIdx.x;
    int block_n = blockIdx.y;
    int m_start = block_m * BLOCK_M;
    int n_start = block_n * BLOCK_N;

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Allocate shared memory for A and B tiles in bf16
    extern __shared__ __nv_bfloat16 shared_mem[];
    __nv_bfloat16* As = shared_mem;                             // size BLOCK_M * BLOCK_K
    __nv_bfloat16* Bs = shared_mem + BLOCK_M * BLOCK_K;        // size BLOCK_K * BLOCK_N

    // Registers for accumulation
    float acc[REG_M][REG_N];
    for(int i = 0; i < REG_M; ++i)
        for(int j = 0; j < REG_N; ++j)
            acc[i][j] = 0.0f;

    // Each thread cooperatively load tiles and compute
    for(int k_start = 0; k_start < K; k_start += BLOCK_K) {
        // Load A tile into shared memory
        int n_elements_A = BLOCK_M * BLOCK_K;
        int tid = ty * THREADS_X + tx;
        int n_threads = THREADS_X * THREADS_Y;
        for(int idx = tid; idx < n_elements_A; idx += n_threads) {
            int m_local = idx / BLOCK_K;
            int k_local = idx % BLOCK_K;
            int m_global = m_start + m_local;
            int k_global = k_start + k_local;
            if(m_global < M && k_global < K) {
                // map m_global -> (a,b)
                int a = m_global / b_dim;
                int b = m_global % b_dim;
                // map k_global -> (x,y)
                int x = k_global / y_dim;
                int y = k_global % y_dim;
                int idxA = ((a * x_dim + x) * b_dim + b) * y_dim + y;
                As[m_local * BLOCK_K + k_local] = A[idxA];
            } else {
                As[m_local * BLOCK_K + k_local] = __float2bfloat16(0.0f);
            }
        }
        // Load B tile into shared memory
        int n_elements_B = BLOCK_K * BLOCK_N;
        for(int idx = tid; idx < n_elements_B; idx += n_threads) {
            int k_local = idx / BLOCK_N;
            int n_local = idx % BLOCK_N;
            int k_global = k_start + k_local;
            int n_global = n_start + n_local;
            if(k_global < K && n_global < N) {
                // map k_global -> (x,y)
                int x = k_global / y_dim;
                int y = k_global % y_dim;
                int c = n_global;
                int idxB = ((x * c_dim + c) * y_dim + y);
                Bs[k_local * BLOCK_N + n_local] = B[idxB];
            } else {
                Bs[k_local * BLOCK_N + n_local] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Register-level blocking
        int row_start = ty * REG_M;
        int col_start = tx * REG_N;
        for(int kk = 0; kk < BLOCK_K; ++kk) {
            // Load A regs
            float a_reg[REG_M];
            for(int i = 0; i < REG_M; ++i) {
                __nv_bfloat16 av = As[(row_start + i) * BLOCK_K + kk];
                a_reg[i] = __bfloat162float(av);
            }
            // Load B regs and compute
            for(int j = 0; j < REG_N; ++j) {
                __nv_bfloat16 bv = Bs[kk * BLOCK_N + (col_start + j)];
                float b_reg = __bfloat162float(bv);
                for(int i = 0; i < REG_M; ++i) {
                    acc[i][j] += a_reg[i] * b_reg;
                }
            }
        }
        __syncthreads();
    }

    // Write back C
    int row_start = ty * REG_M;
    int col_start = tx * REG_N;
    for(int i = 0; i < REG_M; ++i) {
        int m_local = row_start + i;
        int m_global = m_start + m_local;
        if(m_global >= M) continue;
        int a = m_global / b_dim;
        int b = m_global % b_dim;
        for(int j = 0; j < REG_N; ++j) {
            int n_local = col_start + j;
            int n_global = n_start + n_local;
            if(n_global >= N) continue;
            int c = n_global;
            int idxC = ((a * b_dim + b) * c_dim + c);
            C[idxC] = __float2bfloat16(acc[i][j]);
        }
    }
}

// External C wrapper invoking the optimized kernel
extern "C" void high_order_contraction_bf16_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim)
{
    // Flattened sizes
    int M = a_dim * b_dim;
    int N = c_dim;
    dim3 threads(THREADS_X, THREADS_Y);
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M,
              (N + BLOCK_N - 1) / BLOCK_N);
    // Shared memory size
    size_t shared_bytes = (size_t)(BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(__nv_bfloat16);

    high_order_contraction_bf16_kernel_optimized<<<grid, threads, shared_bytes>>>(
        A, B, C,
        a_dim, b_dim, c_dim,
        x_dim, y_dim
    );
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void high_order_contraction_bf16_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_c_elements = a_dim * b_dim * c_dim;
    
    if (idx >= total_c_elements) {
        return;
    }

    int c = idx % c_dim;
    idx = idx / c_dim;
    int b = idx % b_dim;
    int a = idx / b_dim;
    
    float sum = 0.0f;

    for (int x = 0; x < x_dim; x++) {
        for (int y = 0; y < y_dim; y++) {

            int idx_A = ((a * x_dim + x) * b_dim + b) * y_dim + y;
   
            int idx_B = (x * c_dim + c) * y_dim + y;

            sum += __bfloat162float(A[idx_A]) * __bfloat162float(B[idx_B]);
        }
    }
    
    int idx_C = (a * b_dim + b) * c_dim + c;
    
    C[idx_C] = __float2bfloat16(sum);
}

extern "C" void high_order_contraction_bf16_origin(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim) {

    size_t total_c_elements = a_dim * b_dim * c_dim;
    int block_size = 256;
    int grid_size = (total_c_elements + block_size - 1) / block_size;

    high_order_contraction_bf16_kernel<<<grid_size, block_size>>>(
        A, B, C,
        a_dim, b_dim, c_dim,
        x_dim, y_dim
    );
}

// Test case input data structure
typedef struct {
    int a_dim;
    int b_dim;
    int c_dim;
    int x_dim;
    int y_dim;
    __nv_bfloat16* A;
    __nv_bfloat16* B;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<std::vector<int>> abc_list = {
        {32, 32, 32},
        {64, 32, 48}, 
        {128, 128, 64},
        {256, 256, 128}
    };

    std::vector<std::vector<int>> xy_list = {
        {8, 8},
        {16, 16}
    };

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    
    for (int i = 0; i < abc_list.size(); i++) 
    for (int j = 0; j < xy_list.size(); j++) {
        TestCase test_case;
        test_case.a_dim = abc_list[i][0];
        test_case.b_dim = abc_list[i][1];
        test_case.c_dim = abc_list[i][2];
        test_case.x_dim = xy_list[j][0];
        test_case.y_dim = xy_list[j][1];
        
        int A_item = test_case.a_dim * test_case.b_dim * test_case.x_dim * test_case.y_dim;
        int B_item = test_case.x_dim * test_case.y_dim * test_case.c_dim;

        test_case.A = new __nv_bfloat16[A_item];
        test_case.B = new __nv_bfloat16[B_item];
        
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
    printf("Test case size: x_dim: %d, y_dim: %d, a_dim: %d, b_dim: %d, c_dim: %d. Complexity: %ld\n", test_case.x_dim, test_case.y_dim, test_case.a_dim, test_case.b_dim, test_case.c_dim, (long)test_case.a_dim * test_case.b_dim * test_case.c_dim * test_case.x_dim * test_case.y_dim);
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
        const int A_item = test_case.a_dim * test_case.b_dim * test_case.x_dim * test_case.y_dim;
        const int B_item = test_case.x_dim * test_case.y_dim * test_case.c_dim;
        const int C_item = test_case.a_dim * test_case.b_dim * test_case.c_dim;
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
        memcpy(h_A, test_case.A, A_size);
        memcpy(h_B, test_case.B, B_size);

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
            high_order_contraction_bf16_origin(d_A, d_B, d_C, test_case.a_dim, test_case.b_dim, test_case.c_dim, test_case.x_dim, test_case.y_dim);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            high_order_contraction_bf16_optimized(d_A, d_B, d_C_optimized, test_case.a_dim, test_case.b_dim, test_case.c_dim, test_case.x_dim, test_case.y_dim);
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