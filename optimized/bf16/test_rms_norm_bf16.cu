#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>
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
#define WARP_SIZE 32

// First stage: block-level partial sum reduction (no atomics)
__global__ void rms_bf16_partial_kernel_optimized(
    const __nv_bfloat16* input,
    float* d_partial,
    int N) {
    extern __shared__ float sdata[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    // accumulate v*v in register
    for (int i = tid; i < N; i += stride) {
        float v = __bfloat162float(input[i]);
        sum += v * v;
    }
    // intra-warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // warp-level sum in shared memory
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();
    // one warp reduces all warp sums
    if (warpId == 0) {
        int nWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float warpSum = (lane < nWarps) ? sdata[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }
        if (lane == 0) {
            d_partial[blockIdx.x] = warpSum;
        }
    }
}

// Second stage: single-block reduction over partial sums
__global__ void rms_bf16_finalize_kernel_optimized(
    const float* d_partial,
    float* d_rms,
    int numBlocks) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    // accumulate partial sums
    for (int i = tid; i < numBlocks; i += blockDim.x) {
        sum += d_partial[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    // tree-based reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *d_rms = sdata[0];
    }
}

// Normalization kernel unchanged except name
__global__ void rms_norm_bf16_kernel_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const float* d_rms,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float epsilon,
    int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float s = *d_rms;
    float rms = sqrtf(s / (float)N + epsilon);
    for (int i = tid; i < N; i += stride) {
        float x = __bfloat162float(input[i]);
        float x_hat = x / rms;
        float w = __bfloat162float(weight[i]);
        float b = __bfloat162float(bias[i]);
        output[i] = __float2bfloat16(w * x_hat + b);
    }
}

extern "C" void rms_norm_bf16_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const float epsilon,
    const int N) {
    // Allocate device memory for the final rms and block partial sums
    float* d_rms = nullptr;
    float* d_partial = nullptr;
    cudaMalloc(&d_rms, sizeof(float));
    // Determine block count and allocate partials
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    blocks = blocks > 1024 ? 1024 : blocks;
    cudaMalloc(&d_partial, sizeof(float) * blocks);
    cudaMemset(d_rms, 0, sizeof(float));

    // 1) Partial sum kernel
    int sharedMemSize = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);
    rms_bf16_partial_kernel_optimized<<<blocks, threads, sharedMemSize>>>(
        input, d_partial, N);

    // 2) Finalize sum kernel
    int threads2 = 1;
    while (threads2 < blocks) threads2 <<= 1;
    threads2 = threads2 > 1024 ? 1024 : threads2;
    int sharedMem2 = threads2 * sizeof(float);
    rms_bf16_finalize_kernel_optimized<<<1, threads2, sharedMem2>>>(
        d_partial, d_rms, blocks);

    // 3) Normalization kernel
    rms_norm_bf16_kernel_optimized<<<blocks, threads>>>(
        output, input, d_rms, weight, bias, epsilon, N);

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_rms);
    cudaFree(d_partial);
}
// ==== OPTIMIZED KERNEL END ====

__global__ void rms_bf16_kernel(const __nv_bfloat16* input, float* rms, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        float v = __bfloat162float(input[i]);
        local_sum += v * v;
    }

    if (local_sum != 0.0f) {
        atomicAdd(rms, local_sum);
    }
}

__global__ void rms_norm_bf16_kernel(__nv_bfloat16* output,const __nv_bfloat16* input,
    float* d_rms, const __nv_bfloat16* weight, 
    const __nv_bfloat16* bias, float epsilon, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float s   = *d_rms;
    float rms = sqrtf(s / (float)N + epsilon);

    for (int i = tid; i < N; i += stride) {
        float x      = __bfloat162float(input[i]);
        float x_hat  = x / rms;
        output[i]    = __float2bfloat16(__bfloat162float(weight[i]) * x_hat + __bfloat162float(bias[i]));
    }
}

extern "C" void rms_norm_bf16_origin(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const float epsilon,
    const int N) {

    float* d_rms;
    cudaMalloc(&d_rms, sizeof(float));
    cudaMemset(d_rms, 0, sizeof(float));

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;

    rms_bf16_kernel<<<blocks, threadsPerBlock>>>(input, d_rms, N);
    cudaDeviceSynchronize();

    rms_norm_bf16_kernel<<<blocks, threadsPerBlock>>>(output, input, d_rms, weight, bias, epsilon, N);
    cudaDeviceSynchronize();

    cudaFree(d_rms);
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
            rms_norm_bf16_origin(d_output, d_input, d_weight, d_bias, test_case.epsilon, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        auto optimized_kernel = [&]() {
            rms_norm_bf16_optimized(d_output_optimized, d_input, d_weight, d_bias, test_case.epsilon, test_case.N);
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