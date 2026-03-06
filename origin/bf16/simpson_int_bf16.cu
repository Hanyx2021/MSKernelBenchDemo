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

__global__ void simpson_int_bf16_kernel(
    const __nv_bfloat16* y_samples,
    float* partial_sum,
    float a,
    float b,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int stride = gridDim.x * blockDim.x;
    
    float h = (b - a) / (N - 1);
    
    float thread_sum = 0.0f;
    
    for (int i = idx; i < N - 1; i += stride) {
        if (i + 2 >= N) break;
        
        if (i % 2 != 0) continue;

        float f0 = __bfloat162float(y_samples[i]);
        float f1 = __bfloat162float(y_samples[i + 1]);
        float f2 = __bfloat162float(y_samples[i + 2]);
        
        float segment_integral = (h / 3.0f) * (f0 + 4.0f * f1 + f2);
        
        thread_sum += segment_integral;
    }
    
    atomicAdd(partial_sum, thread_sum);
}

__global__ void finalize_simpson_integral_kernel(
    __nv_bfloat16* result,
    float* integral_sum,
    float a,
    float b,
    int N
) {
    float integral_value = *integral_sum;
    *result = __float2bfloat16(integral_value);
}

extern "C" void simpson_int_bf16(
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

    simpson_int_bf16_kernel<<<blocks, threadsPerBlock>>>(
        y_samples, d_integral_sum, a, b, N);

    finalize_simpson_integral_kernel<<<1, 1>>>(
        result, d_integral_sum, a, b, N);

    cudaDeviceSynchronize();
    cudaFree(d_integral_sum);
}