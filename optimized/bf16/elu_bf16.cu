#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Vectorized ELU kernel for BF16 data type (fixed: manual pack/unpack)
extern "C" __global__ void elu_bf16_vector_kernel_optimized(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    int N,
    float alpha) {
    // Reinterpret input/output as vector of two BF16s
    const __nv_bfloat162* __restrict__ in2  = reinterpret_cast<const __nv_bfloat162*>(input);
    __nv_bfloat162*       __restrict__ out2 = reinterpret_cast<__nv_bfloat162*>(out);

    int n2     = N / 2;  // number of full BF16-pairs
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process pairs of elements
    for (int i = tid; i < n2; i += stride) {
        // load pair
        __nv_bfloat162 tmp2 = __ldg(in2 + i);

        // unpack to floats
        float v0 = __bfloat162float(tmp2.x);
        float v1 = __bfloat162float(tmp2.y);

        // apply ELU
        float o0 = (v0 < 0.0f) ? alpha * (expf(v0) - 1.0f) : v0;
        float o1 = (v1 < 0.0f) ? alpha * (expf(v1) - 1.0f) : v1;

        // repack to BF16 pair
        __nv_bfloat162 res;
        res.x = __float2bfloat16(o0);
        res.y = __float2bfloat16(o1);

        // store result
        out2[i] = res;
    }

    // Handle odd tail element if N is odd (only one thread does it)
    if ((N & 1) && tid == 0) {
        int idx = N - 1;
        __nv_bfloat16 tmp = __ldg(input + idx);
        float in_val = __bfloat162float(tmp);
        __nv_bfloat16 out_val = (in_val < 0.0f)
            ? __float2bfloat16(alpha * (expf(in_val) - 1.0f))
            : tmp;
        out[idx] = out_val;
    }
}

// External C wrapper for the optimized ELU kernel
extern "C" void elu_bf16_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const int N) {
    const float alpha    = 1.0f;
    const int block_size = 256;

    // compute grid over pairs
    int n2 = N / 2;
    int grid_size = (n2 + block_size - 1) / block_size;
    if (grid_size == 0) grid_size = 1;

    dim3 block(block_size);
    dim3 grid(grid_size);

    elu_bf16_vector_kernel_optimized<<<grid, block>>>(out, input, N, alpha);
    cudaDeviceSynchronize();
}