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

__global__ void softmax_bf16_kernel_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int N,
    int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const __nv_bfloat16* input_row = input + i * C;
        __nv_bfloat16* out_row = out + i * C;

        float maxval = -FLT_MAX;
        for (int j = 0; j < C; j++) {
            float val = __bfloat162float(input_row[j]);
            if (val > maxval) {
                maxval = val;
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float val = __bfloat162float(input_row[j]);
            float exp_val = expf(val - maxval);
            out_row[j] = __float2bfloat16(exp_val);
            sum += exp_val;
        }
        
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < C; j++) {
            float val = __bfloat162float(out_row[j]);
            out_row[j] = __float2bfloat16(val * inv_sum);
        }
    }
}

extern "C" void softmax_bf16_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int N,
    int C) {
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    softmax_bf16_kernel_optimized<<<grid, block>>>(out, input, N, C);
    cudaDeviceSynchronize();
}

// ==== OPTIMIZED KERNEL END ====

__global__ void softmax_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int N,
    int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const __nv_bfloat16* input_row = input + i * C;
        __nv_bfloat16* out_row = out + i * C;

        float maxval = -FLT_MAX;
        for (int j = 0; j < C; j++) {
            float val = __bfloat162float(input_row[j]);
            if (val > maxval) {
                maxval = val;
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float val = __bfloat162float(input_row[j]);
            float exp_val = expf(val - maxval);
            out_row[j] = __float2bfloat16(exp_val);
            sum += exp_val;
        }
        
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < C; j++) {
            float val = __bfloat162float(out_row[j]);
            out_row[j] = __float2bfloat16(val * inv_sum);
        }
    }
}

extern "C" void softmax_bf16_origin(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int N,
    int C) {
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    softmax_bf16_kernel<<<grid, block>>>(out, input, N, C);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int N;
    int C;
    __nv_bfloat16 *input;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {16, 32, 64, 128};
    std::vector<int> C_list = {1 << 8, 1 << 10, 1 << 12, 1 << 14};

    for (int i = 0; i < N_list.size(); i++) 
        for(int j = 0 ; j < C_list.size(); j++)
        {
            TestCase test_case;
            test_case.N = N_list[i];
            test_case.C = C_list[j];

            // Use fixed seed for reproducibility
            std::random_device rd;
            std::mt19937 rng(rd());  // Random seed for testing
            std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
            
            int input_item = test_case.N * test_case.C;
            test_case.input = new __nv_bfloat16[input_item];
            
            for (int ii = 0; ii < input_item; ii++) {
                test_case.input[ii] = __float2bfloat16(input_dist(rng));
            }
            test_case_list.push_back(test_case);
        }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: N: %d, C: %d. Complexity: %d\n", test_case.N, test_case.C, test_case.N * test_case.C);
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
        const int input_item = test_case.N * test_case.C;
        size_t input_size = input_item * sizeof(__nv_bfloat16);
        size_t output_size = input_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_input = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(output_size);
        __nv_bfloat16* h_output_optimized = (__nv_bfloat16*)malloc(output_size);

        if (!h_input || !h_output || !h_output_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_input, test_case.input, input_size);

        // GPU memory allocation
        __nv_bfloat16 *d_input, *d_output, *d_output_optimized;

        checkCudaError(cudaMalloc((void**)&d_input, input_size), "Allocating d_input");
        checkCudaError(cudaMalloc((void**)&d_output, output_size), "Allocating d_output");
        checkCudaError(cudaMalloc((void**)&d_output_optimized, output_size), "Allocating d_output_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "Copying h_input to d_input");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */
        
        auto origin_kernel = [&]() {
            softmax_bf16_origin(d_output, d_input, test_case.N, test_case.C);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            softmax_bf16_optimized(d_output_optimized, d_input, test_case.N, test_case.C);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
        checkCudaError(cudaMemcpy(h_output_optimized, d_output_optimized, output_size, cudaMemcpyDeviceToHost), "Copying d_output_optimized to h_output_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < input_item; i++) {
            if (!bfloat16_equals(h_output[i], h_output_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_output[i]), __bfloat162float(h_output_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_output_optimized);

        free(h_input);
        free(h_output);
        free(h_output_optimized);
        delete [] test_case.input;
    }

    return 0;
}