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
#include <float.h>


__global__ void softmax_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int N,
    int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const __nv_bfloat16* input_row = input + i * C;
        __nv_bfloat16* out_row = out + i * C;

        float maxval = -FLT_MAX;
        for (int j = 0; j < C; j++) {
            float val = __bfloat162float(input_row[j]);
            if (val > maxval) {
                maxval = val;
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float val = __bfloat162float(input_row[j]);
            float exp_val = expf(val - maxval);
            out_row[j] = __float2bfloat16(exp_val);
            sum += exp_val;
        }
        
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < C; j++) {
            float val = __bfloat162float(out_row[j]);
            out_row[j] = __float2bfloat16(val * inv_sum);
        }
    }
}

extern "C" void softmax_bf16(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int N,
    int C) {
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    softmax_bf16_kernel<<<grid, block>>>(out, input, N, C);
    cudaDeviceSynchronize();
}