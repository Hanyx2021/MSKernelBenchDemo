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
 
__global__ void dot_product_bf16_kernel(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    
    for (int i = tid; i < N; i += stride) {
        float x_val = __bfloat162float(X[i]);
        float y_val = __bfloat162float(Y[i]);
        
        sum += x_val * y_val;
    }
    
    if (sum != 0.0f) {
        atomicAdd(loss, sum);
    }
}

extern "C" void dot_product_bf16(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {
    
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;
    
    dot_product_bf16_kernel<<<blocks, threadsPerBlock>>>(loss, X, Y, N);
    cudaDeviceSynchronize();
}