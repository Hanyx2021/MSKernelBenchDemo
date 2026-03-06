#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device implementation of Swish activation
__device__ float swish_optimized(float x, float beta) {
    return x * (1.0f / (1.0f + expf(-beta * x)));
}

// Optimized kernel with grid-stride loop to eliminate redundant work
__global__ void SwiGLU_kernel_optimized(
    float* output,
    const float* gate_input,
    const float* value_input,
    const float beta,
    const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float gate = gate_input[i];
        float value = value_input[i];
        output[i] = swish_optimized(gate, beta) * value;
    }
}

// External C wrapper for the optimized operator
extern "C" void SwiGLU_optimized(
    float* output,
    const float* gate_input,
    const float* value_input,
    float beta,
    int N) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    SwiGLU_kernel_optimized<<<blocks_per_grid, threads_per_block>>>(
        output, gate_input, value_input, beta, N);
    cudaDeviceSynchronize();
}