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
#include <functional>
#include <cub/cub.cuh>

__global__ void monte_carlo_int_bf16_kernel(
    float* integral_sum,
    const __nv_bfloat16* y_samples,
    float a,
    float b,
    int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    
    for (int i = tid; i < N; i += stride) {
        float y_val = __bfloat162float(y_samples[i]);
        local_sum += y_val;
    }
    
    if (local_sum != 0.0f) {
        atomicAdd(integral_sum, local_sum);
    }
}

__global__ void finalize_integral_kernel(
    float* integral_sum,
    __nv_bfloat16* result,
    float a,
    float b,
    int N) {

    float avg = *integral_sum / static_cast<float>(N);
    float integral_value = (b - a) * avg;

    *result = __float2bfloat16_rn(integral_value);
}

extern "C" void monte_carlo_int_bf16(
    const __nv_bfloat16* y_samples,
    __nv_bfloat16* result,
    float a,
    float b,
    int N) {

    float* d_integral_sum;
    cudaMalloc(&d_integral_sum, sizeof(float));
    cudaMemset(d_integral_sum, 0, sizeof(float));

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;

    monte_carlo_int_bf16_kernel<<<blocks, threadsPerBlock>>>(
        d_integral_sum, y_samples, a, b, N);

    finalize_integral_kernel<<<1, 1>>>(
        d_integral_sum, result, a, b, N);

    cudaDeviceSynchronize();
    cudaFree(d_integral_sum);
}