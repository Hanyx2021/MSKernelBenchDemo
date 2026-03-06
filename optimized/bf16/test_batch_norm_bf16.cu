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
#include <stdint.h>
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
// Kernel for fused statistics (mean & variance) calculation and alpha/beta_prime preparation
// Uses warp-shuffle reduction and manual unrolling for performance
__global__ void batch_norm_fused_stats_prepare_bf16_kernel_optimized(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    float* __restrict__ alpha,
    float* __restrict__ beta_prime,
    int N,
    int C,
    float epsilon) {
    int channel = blockIdx.x;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;        // should be 256
    const int warpSize  = 32;
    const int numWarps  = blockSize / warpSize;  // 8

    // Each thread computes partial sum and sum of squares for this channel
    float local_sum   = 0.0f;
    float local_sqsum = 0.0f;

    // Manual unroll by factor of 2 over N samples
    int i = tid * 2;
    int stride = blockSize * 2;
    while (i + 1 < N) {
        float v0 = __bfloat162float(input[i * C + channel]);
        float v1 = __bfloat162float(input[(i + 1) * C + channel]);
        local_sum   += v0 + v1;
        local_sqsum += v0 * v0 + v1 * v1;
        i += stride;
    }
    // Handle remaining element if N is odd
    if (i < N) {
        float v = __bfloat162float(input[i * C + channel]);
        local_sum   += v;
        local_sqsum += v * v;
    }

    // Intra-warp reduction using shuffle
    unsigned int full_mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum   += __shfl_down_sync(full_mask, local_sum, offset);
        local_sqsum += __shfl_down_sync(full_mask, local_sqsum, offset);
    }

    // Shared memory for inter-warp reduction (one element per warp)
    __shared__ float s_sum[8];
    __shared__ float s_sqsum[8];
    int lane   = tid % warpSize;
    int warpId = tid / warpSize;
    if (lane == 0) {
        s_sum[warpId]   = local_sum;
        s_sqsum[warpId] = local_sqsum;
    }
    __syncthreads();

    // First warp finishes reduction across warps
    if (warpId == 0) {
        float sum   = (lane < numWarps) ? s_sum[lane] : 0.0f;
        float sqsum = (lane < numWarps) ? s_sqsum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum   += __shfl_down_sync(full_mask, sum, offset);
            sqsum += __shfl_down_sync(full_mask, sqsum, offset);
        }
        if (lane == 0) {
            // Compute final statistics and parameters
            float mean    = sum / N;
            float var     = sqsum / N - mean * mean;
            float inv_std = 1.0f / sqrtf(var + epsilon);
            float g = __bfloat162float(gamma[channel]);
            float b = __bfloat162float(beta[channel]);
            float a = g * inv_std;
            float bp = __fmaf_rn(-mean, a, b);
            alpha[channel]      = a;
            beta_prime[channel] = bp;
        }
    }
}

// Kernel for applying batch normalization using precomputed alpha and beta_prime
// Each thread processes two bf16 elements (unrolled)
__global__ void batch_norm_apply_bf16_kernel_optimized(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ alpha,
    const float* __restrict__ beta_prime,
    int N,
    int C) {
    const int blockSize = blockDim.x;  // should be 256
    int total = N * C;
    // Each thread processes two consecutive elements
    int base = (blockIdx.x * blockSize + threadIdx.x) * 2;
    if (base >= total) return;

    // First element
    int idx0 = base;
    int ch0  = idx0 % C;
    float a0 = __ldg(alpha + ch0);
    float b0 = __ldg(beta_prime + ch0);
    float x0 = __bfloat162float(input[idx0]);
    float y0 = __fmaf_rn(a0, x0, b0);
    __nv_bfloat16 o0 = __float2bfloat16(y0);
    output[idx0] = o0;

    // Second element if in bounds
    int idx1 = base + 1;
    if (idx1 < total) {
        int ch1  = idx1 % C;
        float a1 = __ldg(alpha + ch1);
        float b1 = __ldg(beta_prime + ch1);
        float x1 = __bfloat162float(input[idx1]);
        float y1 = __fmaf_rn(a1, x1, b1);
        __nv_bfloat16 o1 = __float2bfloat16(y1);
        output[idx1] = o1;
    }
}

// External C wrapper for the optimized batch normalization operator
extern "C" void batch_norm_bf16_optimized(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    const float          epsilon,
    const int            N,
    const int            C) 
{
    // 1) allocate intermediate alpha and beta_prime
    float *d_alpha = nullptr, *d_beta_prime = nullptr;
    cudaMalloc(&d_alpha,      C * sizeof(float));
    cudaMalloc(&d_beta_prime, C * sizeof(float));

    // 2) launch fused-stats kernel
    const int threadsStats = 256;
    dim3      blockStats(threadsStats);
    dim3      gridStats (C);
    batch_norm_fused_stats_prepare_bf16_kernel_optimized<<<gridStats, blockStats>>>(
        input,
        gamma,
        beta,
        d_alpha,
        d_beta_prime,
        N,
        C,
        epsilon
    );
    cudaDeviceSynchronize();

    // 3) launch apply kernel (each thread does 2 elements)
    const int totalElems   = N * C;
    const int elemsPerTh   = 2;
    const int threadsApply = 256;
    int       pairCount    = (totalElems + elemsPerTh - 1) / elemsPerTh;
    int       blocksApply  = (pairCount + threadsApply - 1) / threadsApply;
    dim3      blockApply(threadsApply);
    dim3      gridApply (blocksApply);
    batch_norm_apply_bf16_kernel_optimized<<<gridApply, blockApply>>>(
        output,
        input,
        d_alpha,
        d_beta_prime,
        N,
        C
    );
    cudaDeviceSynchronize();

    // 4) clean up
    cudaFree(d_alpha);
    cudaFree(d_beta_prime);
}
// ==== OPTIMIZED KERNEL END ====
 
__global__ void batch_norm_stats_bf16_kernel(
    float* means,
    float* variances,
    const __nv_bfloat16* input,
    const int N,
    const int C) {
    
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feature_idx >= C) return;
    
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    for (int sample_idx = 0; sample_idx < N; sample_idx++) {
        int idx = sample_idx * C + feature_idx;
        float val = __bfloat162float(input[idx]);
        sum += val;
        sq_sum += val * val;
    }
    
    float mean = sum / N;
    
    float variance = sq_sum / N - mean * mean;
    
    means[feature_idx] = mean;
    variances[feature_idx] = variance;
}


__global__ void batch_norm_apply_bf16_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const float* means,
    const float* variances,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    const float epsilon,
    const int N,
    const int C) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int global_idx = idx; global_idx < N * C; global_idx += total_threads) {
        int sample_idx = global_idx / C;
        int feature_idx = global_idx % C;
        
        float mean = means[feature_idx];
        float variance = variances[feature_idx];
        
        float scale = __bfloat162float(gamma[feature_idx]);
        float shift = __bfloat162float(beta[feature_idx]);

        float std = sqrtf(variance + epsilon);

        float x = __bfloat162float(input[global_idx]);
        float normalized = (x - mean) / std;
        
        float result_fp32 = scale * normalized + shift;
        output[global_idx] = __float2bfloat16(result_fp32);
    }
}

extern "C" void batch_norm_bf16_origin(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    const float epsilon,
    const int N,
    const int C) {
    
    float* d_means;
    float* d_variances;
    
    cudaMalloc(&d_means, C * sizeof(float));
    cudaMalloc(&d_variances, C * sizeof(float));
    
    const int threadsPerBlock = 256;
    int blocks = (C + threadsPerBlock - 1) / threadsPerBlock;
    
    batch_norm_stats_bf16_kernel<<<blocks, threadsPerBlock>>>(
        d_means, d_variances, input, N, C);
    cudaDeviceSynchronize();
    
    int total_elements = N * C;
    int blocks_apply = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    batch_norm_apply_bf16_kernel<<<blocks_apply, threadsPerBlock>>>(
        output, input, d_means, d_variances, gamma, beta, epsilon, N, C);
    cudaDeviceSynchronize();
    
    cudaFree(d_means);
    cudaFree(d_variances);
}

// Test case input data structure
typedef struct {
    int N;
    int C;
    __nv_bfloat16 *input;
    __nv_bfloat16 *gamma;
    __nv_bfloat16 *beta;
    float epsilon;
} TestCase;

// Function to load test case from hardcoded values
void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 4, 1 << 6, 1 << 8, 1 << 10};
    std::vector<int> C_list = {1 << 4, 1 << 6, 1 << 8, 1 << 10};

    for (int i = 0; i < N_list.size(); i++) 
        for(int j = 0; j < C_list.size(); j++)
        {
            TestCase test_case;
            test_case.N = N_list[i];
            test_case.C = C_list[j];
            test_case.epsilon = 1e-5f;
            
            // Use fixed seed for reproducibility
            std::random_device rd;
            std::mt19937 rng(rd());  // Random seed for testing
            std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
            
            int input_num = test_case.N * test_case.C;
            int param_num = test_case.C;
            test_case.input = new __nv_bfloat16[input_num];
            test_case.gamma = new __nv_bfloat16[param_num];
            test_case.beta = new __nv_bfloat16[param_num];
            
            for (int ii = 0; ii < input_num; ii++) {
                test_case.input[ii] = __float2bfloat16(input_dist(rng));
            }
            for (int ii = 0; ii < param_num; ii++) {
                test_case.gamma[ii] = __float2bfloat16(input_dist(rng));
                test_case.beta[ii] = __float2bfloat16(input_dist(rng));
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
        const int input_num = test_case.N * test_case.C;
        const int param_num = test_case.C;
        size_t input_size = input_num * sizeof(__nv_bfloat16);
        size_t param_size = param_num * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_input = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_gamma = (__nv_bfloat16*)malloc(param_size);
        __nv_bfloat16* h_beta = (__nv_bfloat16*)malloc(param_size);
        __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_output_optimized = (__nv_bfloat16*)malloc(input_size);

        if (!h_input || !h_gamma || !h_beta || !h_output || !h_output_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_input, test_case.input, input_size);
        memcpy(h_gamma, test_case.gamma, param_size);
        memcpy(h_beta, test_case.beta, param_size);

        // GPU memory allocation
        __nv_bfloat16 *d_input, *d_gamma, *d_beta, *d_output, *d_output_optimized;

        checkCudaError(cudaMalloc((void**)&d_input, input_size), "Allocating d_input");
        checkCudaError(cudaMalloc((void**)&d_gamma, param_size), "Allocating d_gamma");
        checkCudaError(cudaMalloc((void**)&d_beta, param_size), "Allocating d_beta");
        checkCudaError(cudaMalloc((void**)&d_output, input_size), "Allocating d_output");
        checkCudaError(cudaMalloc((void**)&d_output_optimized, input_size), "Allocating d_output_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "Copying h_input to d_input");
        checkCudaError(cudaMemcpy(d_gamma, h_gamma, param_size, cudaMemcpyHostToDevice), "Copying h_gamma to d_gamma");
        checkCudaError(cudaMemcpy(d_beta, h_beta, param_size, cudaMemcpyHostToDevice), "Copying h_beta to d_beta");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            batch_norm_bf16_origin(d_output, d_input, d_gamma, d_beta, test_case.epsilon, test_case.N, test_case.C);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            batch_norm_bf16_optimized(d_output_optimized, d_input, d_gamma, d_beta, test_case.epsilon, test_case.N, test_case.C);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
        checkCudaError(cudaMemcpy(h_output_optimized, d_output_optimized, input_size, cudaMemcpyDeviceToHost), "Copying d_output_optimized to h_output_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < input_num; i++) {
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
        cudaFree(d_gamma);
        cudaFree(d_beta);
        cudaFree(d_output);
        cudaFree(d_output_optimized);

        free(h_input);
        free(h_gamma);
        free(h_beta);
        free(h_output);
        free(h_output_optimized);
        delete [] test_case.input;
        delete [] test_case.gamma;
        delete [] test_case.beta;
    }

    return 0;
}