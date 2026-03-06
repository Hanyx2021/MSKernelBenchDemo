#include <cuda.h>
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

__global__ void mean_kernel(const float* input, float* mean_sum, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        local_sum += input[i];
    }

    if (local_sum != 0.0f) {
        atomicAdd(mean_sum, local_sum);
    }
}

__global__ void variance_kernel(const float* input, const float* mean_val, float* var_sum, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float mean = *mean_val;
    float local_var_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        float diff = input[i] - mean;
        local_var_sum += diff * diff;
    }

    if (local_var_sum != 0.0f) {
        atomicAdd(var_sum, local_var_sum);
    }
}

__global__ void layer_norm_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
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
        float normalized = (input[i] - mean) / std;
        output[i] = weight[i] * normalized + bias[i];
    }
}

extern "C" void layer_norm(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
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

    mean_kernel<<<blocks, threadsPerBlock>>>(input, d_mean_sum, N);
    cudaDeviceSynchronize();
    
    float h_mean_sum;
    cudaMemcpy(&h_mean_sum, d_mean_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float mean_val = h_mean_sum / N;
    cudaMemcpy(d_mean_val, &mean_val, sizeof(float), cudaMemcpyHostToDevice);

    variance_kernel<<<blocks, threadsPerBlock>>>(input, d_mean_val, d_var_sum, N);
    cudaDeviceSynchronize();
    
    float h_var_sum;
    cudaMemcpy(&h_var_sum, d_var_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float var_val = h_var_sum / N;
    cudaMemcpy(d_var_val, &var_val, sizeof(float), cudaMemcpyHostToDevice);

    layer_norm_kernel<<<blocks, threadsPerBlock>>>(
        output, input, weight, bias, d_mean_val, d_var_val, epsilon, N);
    cudaDeviceSynchronize();

    cudaFree(d_mean_sum);
    cudaFree(d_var_sum);
    cudaFree(d_mean_val);
    cudaFree(d_var_val);
}