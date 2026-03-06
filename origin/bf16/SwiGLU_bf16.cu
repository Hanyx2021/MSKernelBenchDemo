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

__device__ float swish_bf16(__nv_bfloat16 x_bf16, float beta) {
    float x = __bfloat162float(x_bf16);
    return x * (1.0f / (1.0f + expf(-beta * x)));
}

__global__ void SwiGLU_bf16_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* gate_input,
    const __nv_bfloat16* value_input,
    const float beta,
    const int N) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        __nv_bfloat16 gate_bf16 = gate_input[i];
        __nv_bfloat16 value_bf16 = value_input[i];

        float value_f = __bfloat162float(value_bf16);

        float swish_result = swish_bf16(gate_bf16, beta);
        float result_f = swish_result * value_f;

        output[i] = __float2bfloat16(result_f);
    }
}

extern "C" void SwiGLU_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* gate_input,
    const __nv_bfloat16* value_input,
    float beta,
    int N) {
    
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    SwiGLU_bf16_kernel<<<blocks_per_grid, threads_per_block>>>(output, gate_input, value_input, beta, N);
    cudaDeviceSynchronize();
}