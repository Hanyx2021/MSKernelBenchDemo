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

__global__ void rms_kernel(const float* input, float* rms, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        float v = input[i];
        local_sum += v * v;
    }

    if (local_sum != 0.0f) {
        atomicAdd(rms, local_sum);
    }
}

__global__ void rms_norm_kernel(float* output,const float* input,float* d_rms, const float* weight, const float* bias, float epsilon, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float s   = *d_rms;
    float rms = sqrtf(s / (float)N + epsilon);

    for (int i = tid; i < N; i += stride) {
        float x      = input[i];
        float x_hat  = x / rms;
        output[i]    = weight[i] * x_hat + bias[i];
    }
}

extern "C" void rms_norm(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float epsilon,
    const int N) {

    float* d_rms;
    cudaMalloc(&d_rms, sizeof(float));
    cudaMemset(d_rms, 0, sizeof(float));

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;

    rms_kernel<<<blocks, threadsPerBlock>>>(input, d_rms, N);
    cudaDeviceSynchronize();

    rms_norm_kernel<<<blocks, threadsPerBlock>>>(output, input, d_rms, weight, bias, epsilon, N);
    cudaDeviceSynchronize();

    cudaFree(d_rms);
}