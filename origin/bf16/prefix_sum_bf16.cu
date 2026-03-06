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

__global__ void prefix_sum_bf16_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i <= idx; i++) {
            sum += __bfloat162float(input[i]);
        }
        output[idx] = __float2bfloat16(sum);
    }
}

extern "C" void prefix_sum_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int N)
{
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    
    prefix_sum_bf16_kernel<<<blocks_per_grid, threads_per_block>>>(output, input, N);
    cudaDeviceSynchronize();
}