#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized GELU kernel with vectorized loads and unrolled compute
__global__ void gelu_kernel_optimized(
    float* output,
    const float* input,
    const int N) {
    const int E = 4;
    const float inv_sqrt2 = 0.707106781186547524f;
    int vectorNum = N / E;
    int rem = N - vectorNum * E;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized processing for full float4 chunks
    if (tid < vectorNum) {
        const float4* in4 = reinterpret_cast<const float4*>(input);
        float4* out4 = reinterpret_cast<float4*>(output);
        // Use read-only cache hint for better bandwidth
        float4 x = __ldg(&in4[tid]);
        float4 y;
        y.x = 0.5f * x.x * (1.0f + erff(x.x * inv_sqrt2));
        y.y = 0.5f * x.y * (1.0f + erff(x.y * inv_sqrt2));
        y.z = 0.5f * x.z * (1.0f + erff(x.z * inv_sqrt2));
        y.w = 0.5f * x.w * (1.0f + erff(x.w * inv_sqrt2));
        out4[tid] = y;
    }
    // Tail processing for remaining elements (< E)
    else if (tid < rem) {
        int idx = vectorNum * E + tid;
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + erff(x * inv_sqrt2));
    }
}

extern "C" void gelu_optimized(
    float* output,
    const float* input,
    const int N) {
    const int blockSize = 256;
    int vectorNum = N / 4;
    int rem = N - vectorNum * 4;
    int totalThreads = (vectorNum > rem ? vectorNum : rem);
    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    dim3 block(blockSize, 1, 1);
    dim3 grid(gridSize, 1, 1);
    gelu_kernel_optimized<<<grid, block>>>(output, input, N);
    cudaDeviceSynchronize();
}