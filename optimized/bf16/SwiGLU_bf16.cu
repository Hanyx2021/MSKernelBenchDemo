#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Fast-math optimized swish device function
__device__ float swish_bf16_optimized(__nv_bfloat16 x_bf16, float beta) {
    float x = __bfloat162float(x_bf16);
    // use fast-math intrinsics __expf and __fdividef
    float denom = 1.0f + __expf(-beta * x);
    return x * __fdividef(1.0f, denom);
}

// Optimized kernel with grid-stride loop and __restrict__ qualifiers
__global__ void SwiGLU_bf16_kernel_optimized(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ gate_input,
    const __nv_bfloat16* __restrict__ value_input,
    float beta,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        // load inputs
        __nv_bfloat16 gate_bf16 = gate_input[i];
        __nv_bfloat16 val_bf16 = value_input[i];
        
        // compute swish on gate
        float sw = swish_bf16_optimized(gate_bf16, beta);
        
        // convert value and multiply
        float v = __bfloat162float(val_bf16);
        float result = sw * v;
        
        // store output
        output[i] = __float2bfloat16(result);
    }
}

// External C wrapper calling the optimized kernel
extern "C" void SwiGLU_bf16_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* gate_input,
    const __nv_bfloat16* value_input,
    float beta,
    int N) {
    const int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    SwiGLU_bf16_kernel_optimized<<<blocks_per_grid, threads_per_block>>>(
        output, gate_input, value_input, beta, N);
    cudaDeviceSynchronize();
}