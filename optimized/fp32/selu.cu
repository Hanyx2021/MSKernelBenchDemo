#include <cuda_runtime.h>
#include <math.h>

// SELU activation kernel (optimized)
__global__ void selu_optimized_kernel(
    float* output,
    const float* input,
    int N,
    float alpha,
    float lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = (x < 0.0f)
            ? lambda * alpha * (expf(x) - 1.0f)
            : lambda * x;
    }
}

// External C wrapper for SELU operator (optimized)
extern "C" void selu_optimized(
    float* output,
    const float* input,
    const int N) {
    // Canonical SELU parameters
    const float alpha  = 1.67f;
    const float lambda = 1.0f;

    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    selu_optimized_kernel<<<gridSize, blockSize>>>(output, input, N, alpha, lambda);
    cudaDeviceSynchronize();
}