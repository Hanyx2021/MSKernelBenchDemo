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
#include <cuda_bf16.h>
 
__global__ void cross_entropy_loss_bf16_kernel(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int C) {

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    
    for (int i = tid; i < C; i += stride) {
        float x_val = __bfloat162float(X[i]);
        float y_val = __bfloat162float(Y[i]);
        
        if (y_val != 0.0f) {
            sum += -1.0 * y_val * logf(fmaxf(x_val, 1e-8f));
        }
    }
    
    if (sum != 0.0f) {
        atomicAdd(loss, sum);
    }
}

extern "C" void cross_entropy_loss_bf16(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int C) {

    const int threadsPerBlock = 256;
    int blocks = (C + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;
    
    cross_entropy_loss_bf16_kernel<<<blocks, threadsPerBlock>>>(loss, X, Y, C);
    cudaDeviceSynchronize();
}