#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Fallback reference kernel for bf16 convolution
extern "C" __global__ void conv_2d_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ kernel,
    __nv_bfloat16* __restrict__ output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols);

// External C wrapper for the optimized operator
extern "C" void conv_2d_bf16_optimized(
    const __nv_bfloat16* input,
    const __nv_bfloat16* kernel,
    __nv_bfloat16* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols)
{
    // Compute output dimensions
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    // Always use the simple reference 2D-convolution kernel
    dim3 block(16, 16);
    dim3 grid((output_cols + block.x - 1) / block.x,
              (output_rows + block.y - 1) / block.y);

    // Launch the reference kernel
    conv_2d_bf16_kernel<<<grid, block>>>(
        input, kernel, output,
        input_rows, input_cols,
        kernel_rows, kernel_cols);

    // Synchronize
    cudaDeviceSynchronize();
}
