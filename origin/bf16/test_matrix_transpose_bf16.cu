#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <optional>
#include <algorithm>
#include <random>
#include <cmath>
#include <vector>
#include <tuple>
#include <float.h>

// Simple function to check if two bf16 values are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====

__global__ void matrix_transpose_bf16_kernel_optimized(
    const __nv_bfloat16* A, 
    __nv_bfloat16* B, 
    int M, 
    int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        B[col * M + row] = A[row * N + col];
    }
}
 
extern "C" void matrix_transpose_bf16_optimized(
    const __nv_bfloat16* A, 
    __nv_bfloat16* B, 
    int M, 
    int N) {
    
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    matrix_transpose_bf16_kernel_optimized<<<grid, block>>>(A, B, M, N);
    cudaDeviceSynchronize();
}

// ==== OPTIMIZED KERNEL END ====

__global__ void matrix_transpose_bf16_kernel(
    const __nv_bfloat16* A, 
    __nv_bfloat16* B, 
    int M, 
    int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        B[col * M + row] = A[row * N + col];
    }
}

extern "C" void matrix_transpose_bf16_origin(
    const __nv_bfloat16* A, 
    __nv_bfloat16* B, 
    int M, 
    int N) {
    
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    matrix_transpose_bf16_kernel<<<grid, block>>>(A, B, M, N);
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
    std::vector<int> MN_list = {1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14};

    for (int i = 0; i < MN_list.size(); i++) {
        for (int j = 0; j < MN_list.size(); j++) {
            TestCase test_case;
            test_case.M = MN_list[i];
            test_case.N = MN_list[j];
            
            // Use fixed seed for reproducibility
            std::random_device rd;
            std::mt19937 rng(rd());  // Random seed for testing
            std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
            
            int A_item = test_case.M * test_case.N;
            test_case.A = new __nv_bfloat16[A_item];
            
            for (int k = 0; k < A_item; k++) {
                test_case.A[k] = __float2bfloat16(input_dist(rng));
            }
            test_case_list.push_back(test_case);
        }
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: M: %d, N: %d. Complexity: %ld\n", test_case.M, test_case.N, (long)test_case.M * test_case.N);
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
        const int B_item = test_case.N * test_case.M;
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
            matrix_transpose_bf16_origin(d_A, d_B, test_case.M, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            matrix_transpose_bf16_optimized(d_A, d_B_optimized, test_case.M, test_case.N);
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