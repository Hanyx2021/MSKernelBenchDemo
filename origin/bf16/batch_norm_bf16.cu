#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <optional>
#include <algorithm>
#include <random>
#include <cmath>
#include <vector>
#include <functional>
#include <cub/cub.cuh>

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

extern "C" void batch_norm_bf16(
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