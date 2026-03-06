#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Optimized 1D convolution kernel using shared memory for both BF16 kernel weights
// and BF16 input tiles. Dynamic shared memory holds kernel values and input values as floats.
__global__ void conv_1d_bf16_kernel_optimized(const __nv_bfloat16* input,
                                             const __nv_bfloat16* kernel,
                                             __nv_bfloat16* output,
                                             int input_size,
                                             int kernel_size) {
    extern __shared__ float sh_mem[];
    float* s_kernel = sh_mem;  // [0 .. kernel_size-1]
    float* s_input = sh_mem + kernel_size;  // [kernel_size .. kernel_size + tile_size - 1]

    int t = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int output_size = input_size - kernel_size + 1;

    // Load kernel into shared memory (as float)
    if (t < kernel_size) {
        s_kernel[t] = __bfloat162float(kernel[t]);
    }

    // Cooperative load of input tile: size = blockDim.x + kernel_size - 1
    int tile_size = blockDim.x + kernel_size - 1;
    for (int i = t; i < tile_size; i += blockDim.x) {
        int in_idx = block_start + i;
        if (in_idx < input_size) {
            s_input[i] = __bfloat162float(input[in_idx]);
        } else {
            s_input[i] = 0.0f;
        }
    }
    __syncthreads();

    // Compute convolution for this thread's output element
    int idx = block_start + t;
    if (idx < output_size) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < kernel_size; ++i) {
            acc += s_input[t + i] * s_kernel[i];
        }
        output[idx] = __float2bfloat16(acc);
    }
}

extern "C" void conv_1d_bf16_optimized(const __nv_bfloat16* input,
                                       const __nv_bfloat16* kernel,
                                       __nv_bfloat16* output,
                                       int input_size,
                                       int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    const int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock, 1, 1);
    dim3 grid(blocksPerGrid, 1, 1);

    // Dynamic shared memory: floats for kernel and input tile
    // total floats = kernel_size + (threadsPerBlock + kernel_size - 1)
    size_t shared_mem_bytes = sizeof(float) * (kernel_size + threadsPerBlock + kernel_size - 1);

    conv_1d_bf16_kernel_optimized<<<grid, block, shared_mem_bytes>>>(
        input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
