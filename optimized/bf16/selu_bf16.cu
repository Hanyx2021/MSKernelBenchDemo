#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#define T 4  // Number of BF16 pairs processed per thread

// Optimized SELU kernel for BF16 data with software pipelining, vectorization, and read-only cache
__global__ void selu_bf16_pipeline_kernel_optimized(
    __nv_bfloat16*             output,
    const __nv_bfloat16* __restrict__ input,
    int                         N,
    float                       alpha,
    float                       lambda) {
    int halfN = N >> 1;  // Number of full BF16 pairs

    // Reinterpret as 32-bit word arrays for coalesced vector load/store
    const uint32_t* __restrict__ in32  = reinterpret_cast<const uint32_t*>(input);
    uint32_t*                    out32 = reinterpret_cast<uint32_t*>(output);

    // Compute the starting pair index for this thread
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
    int base_pair   = thread_id * T;

    // Unrolled loop for T pairs per thread
    #pragma unroll
    for (int j = 0; j < T; ++j) {
        int idx_pair = base_pair + j;
        if (idx_pair < halfN) {
            // Prefetch next pair via read-only cache
            uint32_t packed;
            packed = __ldg(in32 + idx_pair);

            // Unpack two BF16 values
            union {
                uint32_t u;
                __nv_bfloat16 b[2];
            } data, result;
            data.u = packed;

            // Compute SELU on element 0
            float f0 = __bfloat162float(data.b[0]);
            float r0 = f0 < 0.0f
                     ? lambda * alpha * (expf(f0) - 1.0f)
                     : lambda * f0;
            result.b[0] = __float2bfloat16(r0);

            // Compute SELU on element 1
            float f1 = __bfloat162float(data.b[1]);
            float r1 = f1 < 0.0f
                     ? lambda * alpha * (expf(f1) - 1.0f)
                     : lambda * f1;
            result.b[1] = __float2bfloat16(r1);

            // Store back as a single 32-bit write
            out32[idx_pair] = result.u;
        }
    }

    // Handle tail element if N is odd
    if ((N & 1) && blockIdx.x == 0 && threadIdx.x == 0) {
        int idx = N - 1;
        float f = __bfloat162float(input[idx]);
        float r = f < 0.0f
                ? lambda * alpha * (expf(f) - 1.0f)
                : lambda * f;
        output[idx] = __float2bfloat16(r);
    }
}

extern "C"
void selu_bf16_optimized(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    const int            N,
    float                alpha  = 1.67f,
    float                lambda = 1.00f) {
    const int block_size       = 128;
    const int pairs_per_thread = T;
    int halfN                  = N >> 1;
    int pairs_per_block        = block_size * pairs_per_thread;
    int grid_size              = (halfN + pairs_per_block - 1) / pairs_per_block;

    // Launch the optimized pipelined SELU kernel
    selu_bf16_pipeline_kernel_optimized<<<grid_size, block_size>>>(
        output, input, N, alpha, lambda);
    cudaDeviceSynchronize();
}
