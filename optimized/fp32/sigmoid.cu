#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized sigmoid kernel with grid-stride loop and software pipelining
__global__ void sigmoid_kernel_optimized(float* out, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i = idx;
    if (i >= N) return;

    // Prefetch the first element
    float val = input[i];
    i += stride;

    // Process remaining elements with double buffering
    while (i < N) {
        float next_val = input[i];
        // Compute sigmoid of val and store
        out[i - stride] = 1.0f / (1.0f + __expf(-val));
        val = next_val;
        i += stride;
    }

    // Handle the last element in the pipeline
    out[i - stride] = 1.0f / (1.0f + __expf(-val));
}

extern "C" void sigmoid_optimized(float* out, const float* input, const int N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 block(blockSize, 1, 1);
    dim3 grid(gridSize, 1, 1);

    sigmoid_kernel_optimized<<<grid, block>>>(out, input, N);
    cudaDeviceSynchronize();
}