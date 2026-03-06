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
#include <functional>

// Simple function to check if two bf16 values are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====

__global__ void matrix_vector_mul_bf16_kernel_optimized(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* x, 
    __nv_bfloat16* y, 
    int M, 
    int N) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            float a = __bfloat162float(A[row * N + col]);
            float b = __bfloat162float(x[col]);
            sum += a * b;
        }
        y[row] = __float2bfloat16(sum);
    }
}
 
extern "C" void matrix_vector_mul_bf16_optimized(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* x, 
    __nv_bfloat16* y, 
    int M, 
    int N) {
    
    const int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    
    matrix_vector_mul_bf16_kernel_optimized<<<grid_size, block_size>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}

// ==== OPTIMIZED KERNEL END ====

__global__ void matrix_vector_mul_bf16_kernel(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* x, 
    __nv_bfloat16* y, 
    int M, 
    int N) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            float a = __bfloat162float(A[row * N + col]);
            float b = __bfloat162float(x[col]);
            sum += a * b;
        }
        y[row] = __float2bfloat16(sum);
    }
}

extern "C" void matrix_vector_mul_bf16_origin(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* x, 
    __nv_bfloat16* y, 
    int M, 
    int N) {
    
    const int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    
    matrix_vector_mul_bf16_kernel<<<grid_size, block_size>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int M;
    int N;
    __nv_bfloat16 *A;
    __nv_bfloat16 *x;
} TestCase;

// Function to load test case from hardcoded values
void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16};
    int M = 2048;

    for (int i = 0; i < N_list.size(); i++) {
        TestCase test_case;
        test_case.M = M;
        test_case.N = N_list[i];  // Larger size for better timing
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int A_item = test_case.M * test_case.N;
        int x_item = test_case.N;
        test_case.A = new __nv_bfloat16[A_item];
        test_case.x = new __nv_bfloat16[x_item];
        
        for (int ii = 0; ii < A_item; ii++) {
            test_case.A[ii] = __float2bfloat16(input_dist(rng));
        }
        for (int ii = 0; ii < x_item; ii++) {
            test_case.x[ii] = __float2bfloat16(input_dist(rng));
        }
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: M: %d, N: %d. Complexity: %ld\n", test_case.M, test_case.N, (long)test_case.M * test_case.N);
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
        const int A_item = test_case.M * test_case.N;
        const int x_item = test_case.N;
        const int y_item = test_case.M;
        size_t A_size = A_item * sizeof(__nv_bfloat16);
        size_t x_size = x_item * sizeof(__nv_bfloat16);
        size_t y_size = y_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_A = (__nv_bfloat16*)malloc(A_size);
        __nv_bfloat16* h_x = (__nv_bfloat16*)malloc(x_size);
        __nv_bfloat16* h_y = (__nv_bfloat16*)malloc(y_size);
        __nv_bfloat16* h_y_optimized = (__nv_bfloat16*)malloc(y_size);

        if (!h_A || !h_x || !h_y || !h_y_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_A, test_case.A, A_size);
        memcpy(h_x, test_case.x, x_size);

        // GPU memory allocation
        __nv_bfloat16 *d_A, *d_x, *d_y, *d_y_optimized;

        checkCudaError(cudaMalloc((void**)&d_A, A_size), "Allocating d_A");
        checkCudaError(cudaMalloc((void**)&d_x, x_size), "Allocating d_x");
        checkCudaError(cudaMalloc((void**)&d_y, y_size), "Allocating d_y");
        checkCudaError(cudaMalloc((void**)&d_y_optimized, y_size), "Allocating d_y_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice), "Copying h_A to d_A");
        checkCudaError(cudaMemcpy(d_x, h_x, x_size, cudaMemcpyHostToDevice), "Copying h_x to d_x");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */
        
        auto origin_kernel = [&]() {
            matrix_vector_mul_bf16_origin(d_A, d_x, d_y, test_case.M, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            matrix_vector_mul_bf16_optimized(d_A, d_x, d_y_optimized, test_case.M, test_case.N);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_y, d_y, y_size, cudaMemcpyDeviceToHost), "Copying d_y to h_y");
        checkCudaError(cudaMemcpy(h_y_optimized, d_y_optimized, y_size, cudaMemcpyDeviceToHost), "Copying d_y_optimized to h_y_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < y_item; i++) {
            if (!bfloat16_equals(h_y[i], h_y_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", 
                       i, __bfloat162float(h_y[i]), __bfloat162float(h_y_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_y_optimized);

        free(h_A);
        free(h_x);
        free(h_y);
        free(h_y_optimized);
        delete [] test_case.A;
        delete [] test_case.x;
    }

    return 0;
}