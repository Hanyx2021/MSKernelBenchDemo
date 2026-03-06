#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized ELU kernel: processes 4 floats per iteration using float4 vector loads/stores
__global__ void elu_kernel_optimized(
    float4* __restrict__ output4,
    const float4* __restrict__ input4,
    int NV,
    float alpha,
    float* __restrict__ output,
    const float* __restrict__ input,
    int rem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < NV; idx += stride) {
        // Load 4 values at once through read-only cache
        float4 v = __ldg(input4 + idx);
        // Compute ELU on each element
        #pragma unroll 1
        for (int i = 0; i < 4; ++i) {
            float x = ((&v.x)[i]);
            float y = x > 0 ? x : alpha * (expf(x) - 1.0f);
            ((&v.x)[i]) = y;
        }
        // Store 4 results at once
        output4[idx] = v;
    }
    // Handle remaining tail elements (N % 4)
    if (rem > 0 && idx == 0) {
        int base = NV * 4;
        for (int i = 0; i < rem; ++i) {
            float x = __ldg(input + base + i);
            float y = x > 0 ? x : alpha * (expf(x) - 1.0f);
            output[base + i] = y;
        }
    }
}

extern "C" void elu_optimized(
    float* output,
    const float* input,
    const int N) {
    // Compute number of vectorized elements and remainder
    const int blockSize = 256;
    int NV = N >> 2;          // number of float4 elements
    int rem = N & 3;          // tail elements
    int gridSizeVec = (NV + blockSize - 1) / blockSize;
    if (gridSizeVec == 0) gridSizeVec = 1;

    dim3 block(blockSize, 1, 1);
    dim3 grid(gridSizeVec, 1, 1);

    // Cast buffers to float4 for vectorized access
    float4* output4 = reinterpret_cast<float4*>(output);
    const float4* input4 = reinterpret_cast<const float4*>(input);
    const float alpha = 1.0f;

    // Launch optimized kernel
    elu_kernel_optimized<<<grid, block>>>(
        output4, input4, NV, alpha,
        output, input, rem);
    cudaDeviceSynchronize();
}
