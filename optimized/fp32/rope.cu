#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Optimized kernel: each thread processes one float2 pair using read-only data cache (__ldg)
__global__ void rope_kernel_optimized(
    float2* __restrict__ output,
    const float2* __restrict__ input,
    const float* __restrict__ d_theta,
    int M,
    int D2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * D2;
    if (idx >= total) return;

    int m = idx / D2;
    int k = idx % D2;

    float2 val = input[idx];
    float theta = __ldg(d_theta + k);
    float angle = theta * m;
    float s, c;
    __sincosf(angle, &s, &c);

    float2 res;
    res.x = val.x * c - val.y * s;
    res.y = val.x * s + val.y * c;
    output[idx] = res;
}

extern "C" void rope_optimized(
    float* output,
    const float* input,
    int M,
    int D,
    float base
) {
    int D2 = D / 2;

    // Precompute theta coefficients on host
    float* h_theta = (float*)malloc(sizeof(float) * D2);
    for (int k = 0; k < D2; ++k) {
        h_theta[k] = powf(base, -2.0f * k / D);
    }

    // Allocate and copy to device global memory
    float* d_theta = nullptr;
    cudaMalloc(&d_theta, sizeof(float) * D2);
    cudaMemcpy(d_theta, h_theta, sizeof(float) * D2, cudaMemcpyHostToDevice);
    free(h_theta);

    int total_pairs = M * D2;
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;

    // Interpret data as float2 vectors for coalesced access
    float2* output2 = reinterpret_cast<float2*>(output);
    const float2* input2 = reinterpret_cast<const float2*>(input);

    // Launch optimized kernel
    rope_kernel_optimized<<<blocks, threads>>>(output2, input2, d_theta, M, D2);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(d_theta);
}
