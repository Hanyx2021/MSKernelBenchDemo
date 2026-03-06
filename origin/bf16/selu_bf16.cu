#include <cuda_bf16.h>
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

__global__ void selu_bf16_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int N,
    float alpha = 1.67,
    float lambda = 1.05) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if(__bfloat162float(input[i]) < 0)
            output[i] = __float2bfloat16(lambda * alpha * (exp(__bfloat162float(input[i])) - 1));
        else
            output[i] = __float2bfloat16(lambda * __bfloat162float(input[i]));
    }
}

extern "C" void selu_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int N,
    float alpha = 1.67,
    float lambda = 1.0) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    selu_bf16_kernel<<<grid, block>>>(output, input, N, alpha, lambda);
    cudaDeviceSynchronize();
}