#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// Optimized kernel: each thread processes elements with a grid-stride loop
__global__ void row_tanh_bf16_kernel_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N) {
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    while (idx < N) {
        float x = __bfloat162float(input[idx]);
        float result = tanhf(x);
        out[idx] = __float2bfloat16(result);
        idx += stride;
    }
}

// External C wrapper with optimized launch configuration
extern "C" void row_tanh_bf16_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N) {
    // Choose block size as a multiple of warp size
    const int block_size = 256;
    // Compute required grid size to cover all elements
    int grid_size = (N + block_size - 1) / block_size;
    
    dim3 block(block_size, 1, 1);
    dim3 grid(grid_size, 1, 1);

    // Launch optimized kernel
    row_tanh_bf16_kernel_optimized<<<grid, block>>>(out, input, N);
    cudaDeviceSynchronize();
}