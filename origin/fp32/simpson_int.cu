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

__global__ void simpson_int_kernel(
    const float* y_samples,
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

        float f0 = y_samples[i];
        float f1 = y_samples[i + 1];
        float f2 = y_samples[i + 2];
        
        float segment_integral = (h / 3.0f) * (f0 + 4.0f * f1 + f2);
        
        thread_sum += segment_integral;
    }
    
    atomicAdd(partial_sum, thread_sum);
}

extern "C" void simpson_int(
    const float* y_samples, 
    float* result, 
    float a, 
    float b, 
    int N) {
    
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    
    simpson_int_kernel<<<blocks_per_grid, threads_per_block>>>(
        y_samples, result, a, b, N
    );
    
    cudaDeviceSynchronize();
}