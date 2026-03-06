#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <optional>
#include <random>
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
// Vectorized ELU kernel for BF16 data type (fixed: manual pack/unpack)
extern "C" __global__ void elu_bf16_vector_kernel_optimized(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    int N,
    float alpha) {
    // Reinterpret input/output as vector of two BF16s
    const __nv_bfloat162* __restrict__ in2  = reinterpret_cast<const __nv_bfloat162*>(input);
    __nv_bfloat162*       __restrict__ out2 = reinterpret_cast<__nv_bfloat162*>(out);

    int n2     = N / 2;  // number of full BF16-pairs
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process pairs of elements
    for (int i = tid; i < n2; i += stride) {
        // load pair
        __nv_bfloat162 tmp2 = __ldg(in2 + i);

        // unpack to floats
        float v0 = __bfloat162float(tmp2.x);
        float v1 = __bfloat162float(tmp2.y);

        // apply ELU
        float o0 = (v0 < 0.0f) ? alpha * (expf(v0) - 1.0f) : v0;
        float o1 = (v1 < 0.0f) ? alpha * (expf(v1) - 1.0f) : v1;

        // repack to BF16 pair
        __nv_bfloat162 res;
        res.x = __float2bfloat16(o0);
        res.y = __float2bfloat16(o1);

        // store result
        out2[i] = res;
    }

    // Handle odd tail element if N is odd (only one thread does it)
    if ((N & 1) && tid == 0) {
        int idx = N - 1;
        __nv_bfloat16 tmp = __ldg(input + idx);
        float in_val = __bfloat162float(tmp);
        __nv_bfloat16 out_val = (in_val < 0.0f)
            ? __float2bfloat16(alpha * (expf(in_val) - 1.0f))
            : tmp;
        out[idx] = out_val;
    }
}

// External C wrapper for the optimized ELU kernel
extern "C" void elu_bf16_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N) {
    const float alpha    = 1.0f;
    const int block_size = 256;

    // compute grid over pairs
    int n2 = N / 2;
    int grid_size = (n2 + block_size - 1) / block_size;
    if (grid_size == 0) grid_size = 1;

    dim3 block(block_size);
    dim3 grid(grid_size);

    elu_bf16_vector_kernel_optimized<<<grid, block>>>(out, input, N, alpha);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

// Original implementation
__global__ void elu_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N,
    float alpha = 1.0) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if(__bfloat162float(input[i]) < 0)
            out[i] = __float2bfloat16(alpha * (exp(__bfloat162float(input[i])) - 1));
        else
            out[i] = input[i];
    }
}

extern "C" void elu_bf16_origin(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N,
    float alpha = 1.0) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    elu_bf16_kernel<<<grid, block>>>(out, input, N, alpha);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int N;
    __nv_bfloat16 *input;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};

    for (int i = 0; i < N_list.size(); i++) {
        TestCase test_case;
        test_case.N = N_list[i];
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int input_item = test_case.N;
        test_case.input = new __nv_bfloat16[input_item];
        
        for (int ii = 0; ii < input_item; ii++) {
            test_case.input[ii] = __float2bfloat16(input_dist(rng));
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
            elu_bf16_origin(d_output, d_input, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            elu_bf16_optimized(d_output_optimized, d_input, test_case.N);
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