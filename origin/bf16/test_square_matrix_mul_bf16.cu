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

__global__ void square_matrix_mul_bf16_kernel_optimized(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, 
    int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < M) {
        float sum = 0.0f;
        for (int k = 0; k < M; k++) {
            float a_val = __bfloat162float(A[row * M + k]);
            float b_val = __bfloat162float(B[k * M + col]);
            sum += a_val * b_val;
        }
        C[row * M + col] = __float2bfloat16(sum);
    }
}

extern "C" void square_matrix_mul_bf16_optimized(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M) {
    dim3 block(16, 16, 1);
    dim3 grid((M + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    square_matrix_mul_bf16_kernel_optimized<<<grid, block>>>(A, B, C, M);
    cudaDeviceSynchronize();
}

// ==== OPTIMIZED KERNEL END ====


__global__ void square_matrix_mul_bf16_kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, 
    int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < M) {
        float sum = 0.0f;
        for (int k = 0; k < M; k++) {
            float a_val = __bfloat162float(A[row * M + k]);
            float b_val = __bfloat162float(B[k * M + col]);
            sum += a_val * b_val;
        }
        C[row * M + col] = __float2bfloat16(sum);
    }
}

extern "C" void square_matrix_mul_bf16_origin(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M) {
    dim3 block(16, 16, 1);
    dim3 grid((M + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    square_matrix_mul_bf16_kernel<<<grid, block>>>(A, B, C, M);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int M;
    __nv_bfloat16 *A;
    __nv_bfloat16 *B;
} TestCase;

// Function to load test case from hardcoded values
void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> M_list = {1 << 4, 1 << 6, 1 << 8, 1 << 10, 1 << 12};

    for (int i = 0; i < M_list.size(); i++) {
        TestCase test_case;
        test_case.M = M_list[i];  // Larger size for better timing
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int item = test_case.M * test_case.M;
        test_case.A = new __nv_bfloat16[item];
        test_case.B = new __nv_bfloat16[item];
        
        for (int ii = 0; ii < item; ii++) {
            test_case.A[ii] = __float2bfloat16(input_dist(rng));
        }
        for (int ii = 0; ii < item; ii++) {
            test_case.B[ii] = __float2bfloat16(input_dist(rng));
        }
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: M: %d. Complexity: %ld\n", test_case.M, (long)test_case.M * test_case.M * test_case.M);
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
        const int item = test_case.M * test_case.M;
        size_t A_size = item * sizeof(__nv_bfloat16);
        size_t B_size = item * sizeof(__nv_bfloat16);
        size_t C_size = item * sizeof(__nv_bfloat16);

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
            square_matrix_mul_bf16_origin(d_A, d_B, d_C, test_case.M);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        // Calling the optimized operator
        auto optimized_kernel = [&]() {
            square_matrix_mul_bf16_optimized(d_A, d_B, d_C_optimized, test_case.M);
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

        for (int i = 0; i < item; i++) {
            if (!bfloat16_equals(h_C[i], h_C_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", 
                       i, __bfloat162float(h_C[i]), __bfloat162float(h_C_optimized[i]));
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
        delete [] test_case.A;
        delete [] test_case.B;
    }

    return 0;
}