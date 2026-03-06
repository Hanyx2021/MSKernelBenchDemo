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

__global__ void rope_kernel(
    float* output,
    const float* input,
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

    float q0 = input[data_offset];
    float q1 = input[data_offset + 1];

    float theta = powf(base, -2.0f * pair / D);
    float m_theta = m * theta;

    float cos_val = cosf(m_theta);
    float sin_val = sinf(m_theta);

    output[data_offset] = q0 * cos_val - q1 * sin_val;
    output[data_offset + 1] = q0 * sin_val + q1 * cos_val;
}

extern "C" void rope(
    float* output,
    const float* input,
    int M,
    int D,
    float base
) {
    int total_pairs = M * (D / 2);
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    rope_kernel<<<blocks, threads>>>(output, input, M, D, base);
    cudaDeviceSynchronize();
}