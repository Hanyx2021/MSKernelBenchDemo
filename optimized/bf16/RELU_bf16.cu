#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Optimized RELU kernel for bfloat16, simple element-wise grid-stride loop
__global__ void RELU_bf16_kernel_optimized(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    int                  N)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements in a grid-stride loop
    for (; idx < N; idx += stride) {
        float x = __bfloat162float(input[idx]);
        float y = x > 0.0f ? x : 0.0f;
        output[idx] = __float2bfloat16(y);
    }
}

extern "C" void RELU_bf16_optimized(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    int                  N)
{
    const int block_size = 256;
    int       grid_size  = (N + block_size - 1) / block_size;
    if (grid_size == 0) grid_size = 1;

    dim3 block(block_size, 1, 1);
    dim3 grid(grid_size, 1, 1);

    RELU_bf16_kernel_optimized<<<grid, block>>>(output, input, N);
    cudaDeviceSynchronize();
}
