#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <optional>
#include <algorithm>
#include <random>
#include <cmath>
#include <vector>
#include <functional>

__global__ void elu_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N,
    float alpha = 1.0) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if(__bfloat162float(input[i]) < 0)
            out[i] = __float2bfloat16(alpha * (exp(__bfloat162float(input[i])) - 1));
        else
            out[i] = input[i];
    }
}

extern "C" void elu_bf16(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N,
    float alpha = 1.0) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    elu_bf16_kernel<<<grid, block>>>(out, input, N, alpha);
    cudaDeviceSynchronize();
}