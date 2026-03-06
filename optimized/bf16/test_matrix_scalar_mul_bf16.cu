#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <functional>
#include <math.h>
#include <optional>
#include <random>
#include <stdint.h>
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
// Vectorized kernel: each thread processes 4 bf16 elements packed in a 64-bit word
__global__ void matrix_scalar_mul_bf16_kernel_optimized(
    const __nv_bfloat16* A, __nv_bfloat16* B,
    __nv_bfloat16 scalar, int M, int N, int N_groups) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_group = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col_group >= N_groups) return;

    // reinterpret A and B as arrays of packed 4 bf16 (=64-bit)
    const uint64_t* A_pack = reinterpret_cast<const uint64_t*>(A);
    uint64_t* B_pack = reinterpret_cast<uint64_t*>(B);

    // load packed data
    uint64_t data = A_pack[row * N_groups + col_group];
    // unpack, multiply, repack
    union { uint64_t u; __nv_bfloat16 v[4]; } tmp;
    tmp.u = data;
    float s = __bfloat162float(scalar);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float x = __bfloat162float(tmp.v[i]);
        tmp.v[i] = __float2bfloat16(x * s);
    }
    // store result
    B_pack[row * N_groups + col_group] = tmp.u;
}

// Tail kernel: handle remaining columns when N % 4 != 0
__global__ void matrix_scalar_mul_bf16_tail_kernel_optimized(
    const __nv_bfloat16* A, __nv_bfloat16* B,
    __nv_bfloat16 scalar, int M, int N, int N_groups, int rem) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= rem) return;

    int idx = row * N + N_groups * 4 + col;
    float x = __bfloat162float(A[idx]);
    float prod = x * __bfloat162float(scalar);
    B[idx] = __float2bfloat16(prod);
}

extern "C" void matrix_scalar_mul_bf16_optimized(
    const __nv_bfloat16* A, __nv_bfloat16* B,
    __nv_bfloat16 scalar, int M, int N) {
    // Compute number of full 4-element groups and remainder
    int N_groups = N / 4;
    int rem = N - N_groups * 4;

    // Launch vectorized kernel for groups of 4 bf16
    if (N_groups > 0) {
        dim3 block(128, 4, 1);
        dim3 grid((N_groups + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y,
                  1);
        matrix_scalar_mul_bf16_kernel_optimized<<<grid, block>>>(
            A, B, scalar, M, N, N_groups);
    }

    // Launch tail kernel for remaining columns
    if (rem > 0) {
        dim3 blockTail(128, 4, 1);
        dim3 gridTail((rem + blockTail.x - 1) / blockTail.x,
                       (M + blockTail.y - 1) / blockTail.y,
                       1);
        matrix_scalar_mul_bf16_tail_kernel_optimized<<<gridTail, blockTail>>>(
            A, B, scalar, M, N, N_groups, rem);
    }

    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void matrix_scalar_mul_bf16_kernel(
    const __nv_bfloat16* A, 
    __nv_bfloat16* B, 
    __nv_bfloat16 scalar, 
    int M, 
    int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float val = __bfloat162float(A[row * N + col]);
        float scaled = val * __bfloat162float(scalar);
        B[row * N + col] = __float2bfloat16(scaled);
    }
}

extern "C" void matrix_scalar_mul_bf16_origin(
    const __nv_bfloat16* A, 
    __nv_bfloat16* B, 
    __nv_bfloat16 scalar, 
    int M, 
    int N) {
    
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    matrix_scalar_mul_bf16_kernel<<<grid, block>>>(A, B, scalar, M, N);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int M;
    int N;
    __nv_bfloat16 *A;
    __nv_bfloat16 scalar;
} TestCase;

// Function to load test case from hardcoded values
void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> M_list = {1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16};
    int N = 2048;

    for (int i = 0; i < M_list.size(); i++) {
        TestCase test_case;
        test_case.M = M_list[i];  // Larger size for better timing
        test_case.N = N;
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int input_item = test_case.M * test_case.N;
        test_case.A = new __nv_bfloat16[input_item];
        
        for (int ii = 0; ii < input_item; ii++) {
            test_case.A[ii] = __float2bfloat16(input_dist(rng));
        }
        test_case.scalar = __float2bfloat16(input_dist(rng));
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: M: %d, N: %d. Complexity: %d\n", test_case.M, test_case.N, test_case.M * test_case.N);
}

// Function to warm up GPU and stabilize frequency
void stabilize_gpu() {
    // Create a dummy kernel to warm up GPU
    __nv_bfloat16 *d_temp;
    cudaMalloc(&d_temp, sizeof(__nv_bfloat16));
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
    
    return total_time;  // Total time for all iterations
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
        const int input_item = test_case.M * test_case.N;
        size_t input_size = input_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_A = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_B = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_B_optimized = (__nv_bfloat16*)malloc(input_size);

        if (!h_A || !h_B || !h_B_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_A, test_case.A, input_size);

        // GPU memory allocation
        __nv_bfloat16 *d_A, *d_B, *d_B_optimized;

        checkCudaError(cudaMalloc((void**)&d_A, input_size), "Allocating d_A");
        checkCudaError(cudaMalloc((void**)&d_B, input_size), "Allocating d_B");
        checkCudaError(cudaMalloc((void**)&d_B_optimized, input_size), "Allocating d_B_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_A, h_A, input_size, cudaMemcpyHostToDevice), "Copying h_A to d_A");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */
        
        auto origin_kernel = [&]() {
            matrix_scalar_mul_bf16_origin(d_A, d_B, test_case.scalar, test_case.M, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            matrix_scalar_mul_bf16_optimized(d_A, d_B_optimized, test_case.scalar, test_case.M, test_case.N);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_B, d_B, input_size, cudaMemcpyDeviceToHost), "Copying d_B to h_B");
        checkCudaError(cudaMemcpy(h_B_optimized, d_B_optimized, input_size, cudaMemcpyDeviceToHost), "Copying d_B_optimized to h_B_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < input_item; i++) {
            if (!bfloat16_equals(h_B[i], h_B_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", 
                       i, __bfloat162float(h_B[i]), __bfloat162float(h_B_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_B_optimized);

        free(h_A);
        free(h_B);
        free(h_B_optimized);
        delete [] test_case.A;
    }

    return 0;
}