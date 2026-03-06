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

__global__ void monte_carlo_int_kernel(
    const float* y_samples,
    float* partial_sum,
    float a,
    float b,
    int N
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float sample_value = y_samples[idx];
    float contribution = (b - a) * sample_value / N;
    
    atomicAdd(partial_sum, contribution);
}

extern "C" void monte_carlo_int(
    const float* y_samples, 
    float* result, 
    float a, 
    float b, 
    int N) {
    
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    monte_carlo_int_kernel<<<blocks_per_grid, threads_per_block>>>(
        y_samples, result, a, b, N
    );

    cudaDeviceSynchronize();
}