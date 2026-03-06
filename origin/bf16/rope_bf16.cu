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
#include <tuple>
#include <float.h>

__global__ void rope_bf16_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int M,
    const int D,
    const float base = 10000.0f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = M * (D / 2);
    if (idx >= total_pairs) return;

    int m = idx / (D / 2);
    int pair = idx % (D / 2);
    int i = pair * 2;
    int data_offset = m * D + i;

    float q0 = __bfloat162float(input[data_offset]);
    float q1 = __bfloat162float(input[data_offset + 1]);

    float theta = powf(base, -2.0f * (float)pair / (float)D);
    float m_theta = (float)m * theta;

    float cos_val = cosf(m_theta);
    float sin_val = sinf(m_theta);

    float out0 = q0 * cos_val - q1 * sin_val;
    float out1 = q0 * sin_val + q1 * cos_val;

    output[data_offset] = __float2bfloat16(out0);
    output[data_offset + 1] = __float2bfloat16(out1);
}

extern "C" void rope_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int M,
    int D,
    float base
) {
    int total_pairs = M * (D / 2);
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    rope_bf16_kernel<<<blocks, threads>>>(output, input, M, D, base);
    cudaDeviceSynchronize();
}