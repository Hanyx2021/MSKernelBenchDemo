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

// Simple function to check if two floats are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====

__global__ void high_order_contraction_bf16_kernel_optimized(
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

extern "C" void high_order_contraction_bf16_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim) {

    size_t total_c_elements = a_dim * b_dim * c_dim;
    int block_size = 256;
    int grid_size = (total_c_elements + block_size - 1) / block_size;

    high_order_contraction_bf16_kernel_optimized<<<grid_size, block_size>>>(
        A, B, C,
        a_dim, b_dim, c_dim,
        x_dim, y_dim
    );
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