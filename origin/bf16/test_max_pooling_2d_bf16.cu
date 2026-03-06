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
__global__ void max_pooling_2d_bf16_kernel_optimized(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_H,
    const int out_W,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h_out >= out_H || w_out >= out_W || n >= N || c >= C) {
        return;
    }
    
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    int h_end = min(h_start + kernel_size, H + padding);
    int w_end = min(w_start + kernel_size, W + padding);
    
    float max_val = -FLT_MAX;
    
    for (int h = max(h_start, 0); h < min(h_end, H); h++) {
        for (int w = max(w_start, 0); w < min(w_end, W); w++) {
            int input_idx = ((n * C + c) * H + h) * W + w;
            float val = __bfloat162float(input[input_idx]);
            
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    
    int output_idx = ((n * C + c) * out_H + h_out) * out_W + w_out;
    output[output_idx] = __float2bfloat16(max_val);
}

extern "C" void max_pooling_2d_bf16_optimized(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int N,
    int C,
    int H,
    int W,
    int kernel_size,
    int stride,
    int padding) {
    
    int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    int out_W = (W + 2 * padding - kernel_size) / stride + 1;
    
    if (out_H <= 0 || out_W <= 0) {
        return;
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (out_W + blockSize.x - 1) / blockSize.x,
        (out_H + blockSize.y - 1) / blockSize.y,
        N * C
    );
    
    max_pooling_2d_bf16_kernel_optimized<<<gridSize, blockSize>>>(
        input, output, N, C, H, W, out_H, out_W, kernel_size, stride, padding);
    
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void max_pooling_2d_bf16_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_H,
    const int out_W,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h_out >= out_H || w_out >= out_W || n >= N || c >= C) {
        return;
    }
    
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    int h_end = min(h_start + kernel_size, H + padding);
    int w_end = min(w_start + kernel_size, W + padding);
    
    float max_val = -FLT_MAX;
    
    for (int h = max(h_start, 0); h < min(h_end, H); h++) {
        for (int w = max(w_start, 0); w < min(w_end, W); w++) {
            int input_idx = ((n * C + c) * H + h) * W + w;
            float val = __bfloat162float(input[input_idx]);
            
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    
    int output_idx = ((n * C + c) * out_H + h_out) * out_W + w_out;
    output[output_idx] = __float2bfloat16(max_val);
}

extern "C" void max_pooling_2d_bf16_origin(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int N,
    int C,
    int H,
    int W,
    int kernel_size,
    int stride,
    int padding) {
    
    int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    int out_W = (W + 2 * padding - kernel_size) / stride + 1;
    
    if (out_H <= 0 || out_W <= 0) {
        return;
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (out_W + blockSize.x - 1) / blockSize.x,
        (out_H + blockSize.y - 1) / blockSize.y,
        N * C
    );
    
    max_pooling_2d_bf16_kernel<<<gridSize, blockSize>>>(
        input, output, N, C, H, W, out_H, out_W, kernel_size, stride, padding);
    
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int N;
    int C;
    int H;
    int W;
    int kernel_size;
    int stride;
    int padding;
    __nv_bfloat16 *input;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<std::vector<int>> input_sizes = {
        {8, 3, 512, 512},
        {4, 3, 224, 224},
        {16, 3, 608, 608},
        {4, 1, 2048, 196}
    };
    
    std::vector<std::vector<int>> pool_params = {
        {2, 1},
        {3, 1},
        {4, 2}
    };
    
    std::vector<int> paddings = {0, 1};

    for (int i = 0; i < input_sizes.size(); i++)
    for(int j = 0; j < pool_params.size(); j++)
    for(int k = 0; k < paddings.size(); k++)
    {
        TestCase test_case;
        test_case.N = input_sizes[i][0];
        test_case.C = input_sizes[i][1];
        test_case.H = input_sizes[i][2];
        test_case.W = input_sizes[i][3];
        test_case.kernel_size = pool_params[j][0];
        test_case.stride = pool_params[j][1];
        test_case.padding = paddings[k];
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);

        test_case.input = new __nv_bfloat16[test_case.N * test_case.C * test_case.H * test_case.W];
        
        for (int ii = 0; ii < test_case.N * test_case.C * test_case.H * test_case.W; ii++) {
            test_case.input[ii] = __float2bfloat16(input_dist(rng));
        }
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    int out_H = (test_case.H + 2 * test_case.padding - test_case.kernel_size) / test_case.stride + 1;
    int out_W = (test_case.W + 2 * test_case.padding - test_case.kernel_size) / test_case.stride + 1;
    long complexity = (long)test_case.N * test_case.C * out_H * out_W * test_case.kernel_size * test_case.kernel_size;
    printf("Test case size: N: %d, C: %d, H: %d, W: %d, kernel_size: %d, stride: %d, padding: %d. Complexity: %ld\n", test_case.N, test_case.C,
           test_case.H, test_case.W, test_case.kernel_size, test_case.stride, test_case.padding, complexity);
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
        const int N = test_case.N;
        const int C = test_case.C;
        const int H = test_case.H;
        const int W = test_case.W;
        const int kernel_size = test_case.kernel_size;
        const int stride = test_case.stride;
        const int padding = test_case.padding;
        const int input_item = N * C * H * W;
        int out_H = (H + 2 * padding - kernel_size) / stride + 1;
        int out_W = (W + 2 * padding - kernel_size) / stride + 1;
        int output_item = N * C * out_H * out_W;
        size_t input_size = input_item * sizeof(__nv_bfloat16);
        size_t output_size = output_item * sizeof(__nv_bfloat16);

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
            max_pooling_2d_bf16_origin(d_input, d_output, N, C, H, W, kernel_size, stride, padding);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            max_pooling_2d_bf16_optimized(d_input, d_output_optimized, N, C, H, W, kernel_size, stride, padding);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "Copying d_output to host");
        checkCudaError(cudaMemcpy(h_output_optimized, d_output_optimized, output_size, cudaMemcpyDeviceToHost), "Copying d_output_optimized to host");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < output_item; i++) {
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