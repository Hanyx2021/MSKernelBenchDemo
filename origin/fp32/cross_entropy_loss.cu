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

__global__ void cross_entropy_loss_kernel(
    float* loss,
    const float* X,
    const float* Y,
    const int C) {

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    
    for (int i = tid; i < C; i += stride) {
        if (Y[i] != 0.0f) {
            sum += -1.0 * Y[i] * logf(fmaxf(X[i], 1e-8f));
        }
    }
    
    if (sum != 0.0f) {
        atomicAdd(loss, sum);
    }
}

extern "C" void cross_entropy_loss(
    float* loss,
    const float* X,
    const float* Y,
    const int C) {
    
    const int threadsPerBlock = 256;
    int blocks = (C + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;
    
    cross_entropy_loss_kernel<<<blocks, threadsPerBlock>>>(loss, X, Y, C);
    cudaDeviceSynchronize();
}