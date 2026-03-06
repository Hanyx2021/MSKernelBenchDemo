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

__global__ void leaky_RELU_bf16_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int N,
    float alpha = 0.01) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = __bfloat162float(input[i]);
        if(val < 0)
            output[i] = __float2bfloat16(alpha * val);
        else
            output[i] = input[i];
    }
}

extern "C" void leaky_RELU_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int N,
    float alpha = 0.01) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    leaky_RELU_bf16_kernel<<<grid, block>>>(output, input, N, alpha);
    cudaDeviceSynchronize();
}