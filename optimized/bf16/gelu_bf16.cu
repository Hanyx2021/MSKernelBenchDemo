#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// Helper for ceiling division
template <typename T>
static inline int ceil_div(T x, T y) {
    return (x + y - 1) / y;
}

// Fast GELU constants
static constexpr float GELU_ALPHA = 0.7978845608028654f;  // sqrt(2/pi)
static constexpr float GELU_BETA  = 0.044715f;

// Optimized GELU kernel mapping one thread to one element
__global__ void gelu_bf16_kernel_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* __restrict__ input,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Load via read-only cache
        __nv_bfloat16 in_val = __ldg(input + idx);
        // Convert to float
        float x = __bfloat162float(in_val);
        // Fast tanh-based GELU approximation
        float x3 = x * x * x;
        float inner = GELU_ALPHA * (x + GELU_BETA * x3);
        float result = 0.5f * x * (1.0f + tanhf(inner));
        // Convert back to bfloat16
        out[idx] = __float2bfloat16(result);
    }
}

extern "C" void gelu_bf16_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N) {
    // Launch configuration
    const int block_size = 256;
    int grid_size = ceil_div(N, block_size);
    dim3 block(block_size, 1, 1);
    dim3 grid(grid_size, 1, 1);

    // Launch optimized kernel
    gelu_bf16_kernel_optimized<<<grid, block>>>(out, input, N);
    cudaDeviceSynchronize();
}