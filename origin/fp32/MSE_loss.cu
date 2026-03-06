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

__global__ void MSE_loss_kernel(
    float* loss,
    const float* X,
    const float* Y,
    const int N) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    for (int i = tid; i < N; i += stride) {
        float diff = X[i] - Y[i];
        sum += diff * diff;
    }

    atomicAdd(loss, sum);
}

__global__ void divide_kernel(float* value, float divisor) {
    *value = *value / divisor;
}

extern "C" void MSE_loss(
    float* loss,
    const float* X,
    const float* Y,
    int N) {

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;
    
    MSE_loss_kernel<<<blocks, threadsPerBlock>>>(loss, X, Y, N);
    
    divide_kernel<<<1, 1>>>(loss, static_cast<float>(N));
    
    cudaDeviceSynchronize();
}