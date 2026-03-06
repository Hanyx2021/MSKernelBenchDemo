#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Maximum supported kernel size for constant memory
#define MAX_KERNEL_SIZE 256

// Kernel coefficients loaded into constant memory for fast broadcast
__constant__ float const_kernel[MAX_KERNEL_SIZE];

// 1D convolution kernel with shared-memory tiling (optimized fix)
extern "C" __global__ void conv_1d_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int kernel_size) {
    extern __shared__ float shared_input[];

    int tid = threadIdx.x;
    int block_input_start = blockIdx.x * blockDim.x;
    int in_idx = block_input_start + tid;

    // 1) Load main tile
    if (in_idx < input_size) {
        shared_input[tid] = input[in_idx];
    } else {
        shared_input[tid] = 0.0f;
    }

    // 2) Load halo region
    if (tid < kernel_size - 1) {
        int halo_idx = block_input_start + blockDim.x + tid;
        if (halo_idx < input_size) {
            shared_input[blockDim.x + tid] = input[halo_idx];
        } else {
            shared_input[blockDim.x + tid] = 0.0f;
        }
    }

    __syncthreads();

    // Compute this thread's global output index
    int global_out_idx = block_input_start + tid;
    int output_size = input_size - kernel_size + 1;

    // 3) Only in‐range threads perform the convolution & write
    if (global_out_idx < output_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            sum += shared_input[tid + k] * const_kernel[k];
        }
        output[global_out_idx] = sum;
    }
}

// External C wrapper for the optimized 1D convolution (unchanged signature)
extern "C" void conv_1d_optimized(
    const float* input,
    const float* kernel,
    float* output,
    int input_size,
    int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    const int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    // Copy filter coefficients into constant memory
    cudaMemcpyToSymbol(const_kernel, kernel, kernel_size * sizeof(float));

    // Shared memory size: (threadsPerBlock + kernel_size - 1) floats
    size_t shared_mem_bytes = (threadsPerBlock + kernel_size - 1) * sizeof(float);

    dim3 block(threadsPerBlock, 1, 1);
    dim3 grid(blocksPerGrid, 1, 1);

    conv_1d_kernel_optimized<<<grid, block, shared_mem_bytes>>>(
        input, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}