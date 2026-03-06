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

__global__ void dot_product_kernel(
    float* loss,
    const float* X,
    const float* Y,
    const int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    
    for (int i = tid; i < N; i += stride) {
        if (Y[i] != 0.0f) {
            sum += X[i] * Y[i];
        }
    }
    
    if (sum != 0.0f) {
        atomicAdd(loss, sum);
    }
}

extern "C" void dot_product(float* loss, const float* X, const float* Y, const int N) {

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;
    
    dot_product_kernel<<<blocks, threadsPerBlock>>>(loss, X, Y, N);
    cudaDeviceSynchronize();
}