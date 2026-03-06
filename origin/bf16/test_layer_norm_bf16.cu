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
#include <cub/cub.cuh>

// Simple function to check if two bf16 values are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====
__global__ void mean_bf16_kernel_optimized(const __nv_bfloat16* input, float* mean_sum, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        float val = __bfloat162float(input[i]);
        local_sum += val;
    }

    if (local_sum != 0.0f) {
        atomicAdd(mean_sum, local_sum);
    }
}

__global__ void variance_bf16_kernel_optimized(
    const __nv_bfloat16* input, 
    const float* mean_val, 
    float* var_sum, 
    int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float mean = *mean_val;
    float local_var_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        float val = __bfloat162float(input[i]);
        float diff = val - mean;
        local_var_sum += diff * diff;
    }

    if (local_var_sum != 0.0f) {
        atomicAdd(var_sum, local_var_sum);
    }
}

__global__ void layer_norm_bf16_kernel_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const float* mean_val,
    const float* var_val,
    float epsilon,
    int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float mean = *mean_val;
    float variance = *var_val;
    
    float std = sqrtf(variance + epsilon);

    for (int i = tid; i < N; i += stride) {
        float x_float = __bfloat162float(input[i]);
        float w_float = __bfloat162float(weight[i]);
        float b_float = __bfloat162float(bias[i]);
        
        float normalized = (x_float - mean) / std;
        float result = w_float * normalized + b_float;
        
        output[i] = __float2bfloat16(result);
    }
}

extern "C" void layer_norm_bf16_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const float epsilon,
    const int N) {
    
    float* d_mean_sum;
    float* d_var_sum;
    float* d_mean_val;
    float* d_var_val;
    
    cudaMalloc(&d_mean_sum, sizeof(float));
    cudaMalloc(&d_var_sum, sizeof(float));
    cudaMalloc(&d_mean_val, sizeof(float));
    cudaMalloc(&d_var_val, sizeof(float));
    
    float zero = 0.0f;
    cudaMemcpy(d_mean_sum, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_sum, &zero, sizeof(float), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;

    mean_bf16_kernel_optimized<<<blocks, threadsPerBlock>>>(input, d_mean_sum, N);
    cudaDeviceSynchronize();
    
    float h_mean_sum;
    cudaMemcpy(&h_mean_sum, d_mean_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float mean_val = h_mean_sum / N;
    cudaMemcpy(d_mean_val, &mean_val, sizeof(float), cudaMemcpyHostToDevice);

    variance_bf16_kernel_optimized<<<blocks, threadsPerBlock>>>(input, d_mean_val, d_var_sum, N);
    cudaDeviceSynchronize();
    
    float h_var_sum;
    cudaMemcpy(&h_var_sum, d_var_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float var_val = h_var_sum / N;
    cudaMemcpy(d_var_val, &var_val, sizeof(float), cudaMemcpyHostToDevice);

    layer_norm_bf16_kernel_optimized<<<blocks, threadsPerBlock>>>(
        output, input, weight, bias, d_mean_val, d_var_val, epsilon, N);
    cudaDeviceSynchronize();

    cudaFree(d_mean_sum);
    cudaFree(d_var_sum);
    cudaFree(d_mean_val);
    cudaFree(d_var_val);
}
// ==== OPTIMIZED KERNEL END ====
 
__global__ void mean_bf16_kernel(const __nv_bfloat16* input, float* mean_sum, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        float val = __bfloat162float(input[i]);
        local_sum += val;
    }

    if (local_sum != 0.0f) {
        atomicAdd(mean_sum, local_sum);
    }
}

__global__ void variance_bf16_kernel(
    const __nv_bfloat16* input, 
    const float* mean_val, 
    float* var_sum, 
    int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float mean = *mean_val;
    float local_var_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        float val = __bfloat162float(input[i]);
        float diff = val - mean;
        local_var_sum += diff * diff;
    }

    if (local_var_sum != 0.0f) {
        atomicAdd(var_sum, local_var_sum);
    }
}

__global__ void layer_norm_bf16_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const float* mean_val,
    const float* var_val,
    float epsilon,
    int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float mean = *mean_val;
    float variance = *var_val;
    
    float std = sqrtf(variance + epsilon);

    for (int i = tid; i < N; i += stride) {
        float x_float = __bfloat162float(input[i]);
        float w_float = __bfloat162float(weight[i]);
        float b_float = __bfloat162float(bias[i]);
        
        float normalized = (x_float - mean) / std;
        float result = w_float * normalized + b_float;
        
        output[i] = __float2bfloat16(result);
    }
}

extern "C" void layer_norm_bf16_origin(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const float epsilon,
    const int N) {
    
    float* d_mean_sum;
    float* d_var_sum;
    float* d_mean_val;
    float* d_var_val;
    
    cudaMalloc(&d_mean_sum, sizeof(float));
    cudaMalloc(&d_var_sum, sizeof(float));
    cudaMalloc(&d_mean_val, sizeof(float));
    cudaMalloc(&d_var_val, sizeof(float));
    
    float zero = 0.0f;
    cudaMemcpy(d_mean_sum, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_sum, &zero, sizeof(float), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;

    mean_bf16_kernel<<<blocks, threadsPerBlock>>>(input, d_mean_sum, N);
    cudaDeviceSynchronize();
    
    float h_mean_sum;
    cudaMemcpy(&h_mean_sum, d_mean_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float mean_val = h_mean_sum / N;
    cudaMemcpy(d_mean_val, &mean_val, sizeof(float), cudaMemcpyHostToDevice);

    variance_bf16_kernel<<<blocks, threadsPerBlock>>>(input, d_mean_val, d_var_sum, N);
    cudaDeviceSynchronize();
    
    float h_var_sum;
    cudaMemcpy(&h_var_sum, d_var_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float var_val = h_var_sum / N;
    cudaMemcpy(d_var_val, &var_val, sizeof(float), cudaMemcpyHostToDevice);

    layer_norm_bf16_kernel<<<blocks, threadsPerBlock>>>(
        output, input, weight, bias, d_mean_val, d_var_val, epsilon, N);
    cudaDeviceSynchronize();

    cudaFree(d_mean_sum);
    cudaFree(d_var_sum);
    cudaFree(d_mean_val);
    cudaFree(d_var_val);
}

// Test case input data structure
typedef struct {
    int N;
    __nv_bfloat16 *input;
    __nv_bfloat16 *weight;
    __nv_bfloat16 *bias;
    float epsilon;
} TestCase;

// Function to load test case from hardcoded values
void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10};

    for (int i = 0; i < N_list.size(); i++) {
        TestCase test_case;
        test_case.N = N_list[i];
        test_case.epsilon = 1e-5f;
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int item_count = test_case.N;
        test_case.input = new __nv_bfloat16[item_count];
        test_case.weight = new __nv_bfloat16[item_count];
        test_case.bias = new __nv_bfloat16[item_count];
        
        for (int ii = 0; ii < item_count; ii++) {
            test_case.input[ii] = __float2bfloat16(input_dist(rng));
            test_case.weight[ii] = __float2bfloat16(input_dist(rng));
            test_case.bias[ii] = __float2bfloat16(input_dist(rng));
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
        const int item_count = test_case.N;
        size_t data_size = item_count * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_input = (__nv_bfloat16*)malloc(data_size);
        __nv_bfloat16* h_weight = (__nv_bfloat16*)malloc(data_size);
        __nv_bfloat16* h_bias = (__nv_bfloat16*)malloc(data_size);
        __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(data_size);
        __nv_bfloat16* h_output_optimized = (__nv_bfloat16*)malloc(data_size);

        if (!h_input || !h_weight || !h_bias || !h_output || !h_output_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_input, test_case.input, data_size);
        memcpy(h_weight, test_case.weight, data_size);
        memcpy(h_bias, test_case.bias, data_size);

        // GPU memory allocation
        __nv_bfloat16 *d_input, *d_weight, *d_bias, *d_output, *d_output_optimized;

        checkCudaError(cudaMalloc((void**)&d_input, data_size), "Allocating d_input");
        checkCudaError(cudaMalloc((void**)&d_weight, data_size), "Allocating d_weight");
        checkCudaError(cudaMalloc((void**)&d_bias, data_size), "Allocating d_bias");
        checkCudaError(cudaMalloc((void**)&d_output, data_size), "Allocating d_output");
        checkCudaError(cudaMalloc((void**)&d_output_optimized, data_size), "Allocating d_output_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice), "Copying h_input to d_input");
        checkCudaError(cudaMemcpy(d_weight, h_weight, data_size, cudaMemcpyHostToDevice), "Copying h_weight to d_weight");
        checkCudaError(cudaMemcpy(d_bias, h_bias, data_size, cudaMemcpyHostToDevice), "Copying h_bias to d_bias");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            layer_norm_bf16_origin(d_output, d_input, d_weight, d_bias, test_case.epsilon, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            layer_norm_bf16_optimized(d_output_optimized, d_input, d_weight, d_bias, test_case.epsilon, test_case.N);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
        checkCudaError(cudaMemcpy(h_output_optimized, d_output_optimized, data_size, cudaMemcpyDeviceToHost), "Copying d_output_optimized to h_output_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < item_count; i++) {
            if (!bfloat16_equals(h_output[i], h_output_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", 
                       i, __bfloat162float(h_output[i]), __bfloat162float(h_output_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_bias);
        cudaFree(d_output);
        cudaFree(d_output_optimized);

        free(h_input);
        free(h_weight);
        free(h_bias);
        free(h_output);
        free(h_output_optimized);
        delete [] test_case.input;
        delete [] test_case.weight;
        delete [] test_case.bias;
    }

    return 0;
}