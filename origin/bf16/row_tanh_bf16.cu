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


__global__ void row_tanh_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N) {
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
        const float x = __bfloat162float(input[idx]);
        float result = tanhf(x);
        out[idx] = __float2bfloat16(result);
    }
}

extern "C" void row_tanh_bf16(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N) {
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    row_tanh_bf16_kernel<<<grid, block>>>(out, input, N);
    cudaDeviceSynchronize();
}