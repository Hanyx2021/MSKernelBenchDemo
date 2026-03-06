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

__device__ float swish(float x, float beta) {
    return x * (1.0f / (1.0f + expf(-beta * x)));
}

__global__ void SwiGLU_kernel(
    float* output,
    const float* gate_input,
    const float* value_input,
    const float beta, 
    const int N) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float gate = gate_input[i];
        float value = value_input[i];
        output[i] = swish(gate, beta) * value;
    }
}

extern "C" void SwiGLU(
    float* output,
    const float* gate_input,
    const float* value_input,
    float beta,
    int N) {
    
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    SwiGLU_kernel<<<blocks_per_grid, threads_per_block>>>(output, gate_input, value_input, beta, N);
    cudaDeviceSynchronize();
}