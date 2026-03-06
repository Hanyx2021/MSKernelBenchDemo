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

__device__ float swish_bf16_origin_optimized(__nv_bfloat16 x_bf16, float beta) {
    float x = __bfloat162float(x_bf16);
    return x * (1.0f / (1.0f + expf(-beta * x)));
}

__global__ void SwiGLU_bf16_kernel_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* gate_input,
    const __nv_bfloat16* value_input,
    const float beta,
    const int N) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        __nv_bfloat16 gate_bf16 = gate_input[i];
        __nv_bfloat16 value_bf16 = value_input[i];

        float value_f = __bfloat162float(value_bf16);

        float swish_result = swish_bf16_origin_optimized(gate_bf16, beta);
        float result_f = swish_result * value_f;

        output[i] = __float2bfloat16(result_f);
    }
}

extern "C" void SwiGLU_bf16_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* gate_input,
    const __nv_bfloat16* value_input,
    float beta,
    int N) {
    
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    SwiGLU_bf16_kernel_optimized<<<blocks_per_grid, threads_per_block>>>(output, gate_input, value_input, beta, N);
    cudaDeviceSynchronize();
}

// ==== OPTIMIZED KERNEL END ====

__device__ float swish_bf16_origin(__nv_bfloat16 x_bf16, float beta) {
    float x = __bfloat162float(x_bf16);
    return x * (1.0f / (1.0f + expf(-beta * x)));
}

__global__ void SwiGLU_bf16_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* gate_input,
    const __nv_bfloat16* value_input,
    const float beta,
    const int N) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        __nv_bfloat16 gate_bf16 = gate_input[i];
        __nv_bfloat16 value_bf16 = value_input[i];

        float value_f = __bfloat162float(value_bf16);

        float swish_result = swish_bf16_origin(gate_bf16, beta);
        float result_f = swish_result * value_f;

        output[i] = __float2bfloat16(result_f);
    }
}

extern "C" void SwiGLU_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* gate_input,
    const __nv_bfloat16* value_input,
    float beta,
    int N) {
    
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    SwiGLU_bf16_kernel<<<blocks_per_grid, threads_per_block>>>(output, gate_input, value_input, beta, N);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int N;
    float beta;
    __nv_bfloat16 *gate_input;
    __nv_bfloat16* value_input;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};

    for (int i = 0; i < N_list.size(); i++) {
        TestCase test_case;
        test_case.N = N_list[i];
        test_case.beta = 1.0;
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int input_item = test_case.N;
        test_case.gate_input = new __nv_bfloat16[input_item];
        test_case.value_input = new __nv_bfloat16[input_item];
        
        for (int ii = 0; ii < input_item; ii++) {
            test_case.gate_input[ii] = __float2bfloat16(input_dist(rng));
            test_case.value_input[ii] = __float2bfloat16(input_dist(rng));
        }
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: N: %d. Complexity: %d\n", test_case.N, test_case.N);
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
        const int input_item = test_case.N;
        size_t input_size = input_item * sizeof(__nv_bfloat16);
        size_t output_size = input_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_gate_input = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_value_input = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(output_size);
        __nv_bfloat16* h_output_optimized = (__nv_bfloat16*)malloc(output_size);

        if (!h_gate_input || !h_value_input || !h_output || !h_output_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_gate_input, test_case.gate_input, input_size);
        memcpy(h_value_input, test_case.value_input, input_size);

        // GPU memory allocation
        __nv_bfloat16 *d_gate_input, *d_value_input, *d_output, *d_output_optimized;

        checkCudaError(cudaMalloc((void**)&d_gate_input, input_size), "Allocating d_gate_input");
        checkCudaError(cudaMalloc((void**)&d_value_input, input_size), "Allocating d_value_input");
        checkCudaError(cudaMalloc((void**)&d_output, output_size), "Allocating d_output");
        checkCudaError(cudaMalloc((void**)&d_output_optimized, output_size), "Allocating d_output_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_gate_input, h_gate_input, input_size, cudaMemcpyHostToDevice), "Copying h_gate_input to d_gate_input");
        checkCudaError(cudaMemcpy(d_value_input, h_value_input, input_size, cudaMemcpyHostToDevice), "Copying h_value_input to d_value_input");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */
        
        auto origin_kernel = [&]() {
            SwiGLU_bf16(d_output, d_gate_input, d_value_input, test_case.beta, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            SwiGLU_bf16_optimized(d_output_optimized, d_gate_input, d_value_input, test_case.beta, test_case.N);
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
        cudaFree(d_gate_input);
        cudaFree(d_value_input);
        cudaFree(d_output);
        cudaFree(d_output_optimized);

        free(h_gate_input);
        free(h_value_input);
        free(h_output);
        free(h_output_optimized);
        delete [] test_case.gate_input;
        delete [] test_case.value_input;
    }

    return 0;
}