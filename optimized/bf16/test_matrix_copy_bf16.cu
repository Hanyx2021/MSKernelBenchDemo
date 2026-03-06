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


// Simple function to check if two bf16 values are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====
// Vectorized 2x bf16 per thread copy kernel
__global__ void matrix_copy_bf16_kernel_optimized(const __nv_bfloat16* A,
                                                  __nv_bfloat16* B,
                                                  int M,
                                                  int N) {
    size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_vec = (total + 1) / 2;

    if (idx < total_vec) {
        // Reinterpret pointers as 32-bit to copy two bf16s at once
        const uint32_t* A32 = reinterpret_cast<const uint32_t*>(A);
        uint32_t*       B32 = reinterpret_cast<uint32_t*>(B);

        // Handle odd tail: last thread and odd total
        if ((idx == total_vec - 1) && (total & 1)) {
            size_t pos = total - 1;
            B[pos] = A[pos];
        } else {
            // Vectorized copy: two bf16 elements per uint32 load/store
            B32[idx] = A32[idx];
        }
    }
}

// External C wrapper for the optimized kernel
extern "C" void matrix_copy_bf16_optimized(const __nv_bfloat16* A,
                                            __nv_bfloat16* B,
                                            int M,
                                            int N) {
    size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
    size_t total_vec = (total + 1) / 2;
    const int threads = 256;
    int blocks = static_cast<int>((total_vec + threads - 1) / threads);

    matrix_copy_bf16_kernel_optimized<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void matrix_copy_bf16_kernel(__nv_bfloat16* A, __nv_bfloat16* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        B[row * N + col] = A[row * N + col];
    }
}

extern "C" void matrix_copy_bf16_origin(__nv_bfloat16* A, __nv_bfloat16* B, int M, int N) {
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    matrix_copy_bf16_kernel<<<grid, block>>>(A, B, M, N);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int M;
    int N;
    __nv_bfloat16 *A;
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
        
        int A_item = test_case.M * test_case.N;
        test_case.A = new __nv_bfloat16[A_item];
        
        for (int ii = 0; ii < A_item; ii++) {
            test_case.A[ii] = __float2bfloat16(input_dist(rng));
        }
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
        const int A_item = test_case.M * test_case.N;
        const int B_item = test_case.M * test_case.N;
        size_t A_size = A_item * sizeof(__nv_bfloat16);
        size_t B_size = B_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_A = (__nv_bfloat16*)malloc(A_size);
        __nv_bfloat16* h_B = (__nv_bfloat16*)malloc(B_size);
        __nv_bfloat16* h_B_optimized = (__nv_bfloat16*)malloc(B_size);

        if (!h_A || !h_B || !h_B_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_A, test_case.A, A_size);

        // GPU memory allocation
        __nv_bfloat16 *d_A, *d_B, *d_B_optimized;

        checkCudaError(cudaMalloc((void**)&d_A, A_size), "Allocating d_A");
        checkCudaError(cudaMalloc((void**)&d_B, B_size), "Allocating d_B");
        checkCudaError(cudaMalloc((void**)&d_B_optimized, B_size), "Allocating d_B_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice), "Copying h_A to d_A");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            matrix_copy_bf16_origin(d_A, d_B, test_case.M, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            matrix_copy_bf16_optimized(d_A, d_B_optimized, test_case.M, test_case.N);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_B, d_B, B_size, cudaMemcpyDeviceToHost), "Copying d_B to h_B");
        checkCudaError(cudaMemcpy(h_B_optimized, d_B_optimized, B_size, cudaMemcpyDeviceToHost), "Copying d_B_optimized to h_B_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < B_item; i++) {
            if (!bfloat16_equals(h_B[i], h_B_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_B[i]), __bfloat162float(h_B_optimized[i]));
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