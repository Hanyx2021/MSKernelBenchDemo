#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// Kernel for fused statistics (mean & variance) calculation and alpha/beta_prime preparation
// Uses warp-shuffle reduction and manual unrolling for performance
__global__ void batch_norm_fused_stats_prepare_bf16_kernel_optimized(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    float* __restrict__ alpha,
    float* __restrict__ beta_prime,
    int N,
    int C,
    float epsilon) {
    int channel = blockIdx.x;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;        // should be 256
    const int warpSize  = 32;
    const int numWarps  = blockSize / warpSize;  // 8

    // Each thread computes partial sum and sum of squares for this channel
    float local_sum   = 0.0f;
    float local_sqsum = 0.0f;

    // Manual unroll by factor of 2 over N samples
    int i = tid * 2;
    int stride = blockSize * 2;
    while (i + 1 < N) {
        float v0 = __bfloat162float(input[i * C + channel]);
        float v1 = __bfloat162float(input[(i + 1) * C + channel]);
        local_sum   += v0 + v1;
        local_sqsum += v0 * v0 + v1 * v1;
        i += stride;
    }
    // Handle remaining element if N is odd
    if (i < N) {
        float v = __bfloat162float(input[i * C + channel]);
        local_sum   += v;
        local_sqsum += v * v;
    }

    // Intra-warp reduction using shuffle
    unsigned int full_mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum   += __shfl_down_sync(full_mask, local_sum, offset);
        local_sqsum += __shfl_down_sync(full_mask, local_sqsum, offset);
    }

    // Shared memory for inter-warp reduction (one element per warp)
    __shared__ float s_sum[8];
    __shared__ float s_sqsum[8];
    int lane   = tid % warpSize;
    int warpId = tid / warpSize;
    if (lane == 0) {
        s_sum[warpId]   = local_sum;
        s_sqsum[warpId] = local_sqsum;
    }
    __syncthreads();

    // First warp finishes reduction across warps
    if (warpId == 0) {
        float sum   = (lane < numWarps) ? s_sum[lane] : 0.0f;
        float sqsum = (lane < numWarps) ? s_sqsum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum   += __shfl_down_sync(full_mask, sum, offset);
            sqsum += __shfl_down_sync(full_mask, sqsum, offset);
        }
        if (lane == 0) {
            // Compute final statistics and parameters
            float mean    = sum / N;
            float var     = sqsum / N - mean * mean;
            float inv_std = 1.0f / sqrtf(var + epsilon);
            float g = __bfloat162float(gamma[channel]);
            float b = __bfloat162float(beta[channel]);
            float a = g * inv_std;
            float bp = __fmaf_rn(-mean, a, b);
            alpha[channel]      = a;
            beta_prime[channel] = bp;
        }
    }
}

// Kernel for applying batch normalization using precomputed alpha and beta_prime
// Each thread processes two bf16 elements (unrolled)
__global__ void batch_norm_apply_bf16_kernel_optimized(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ alpha,
    const float* __restrict__ beta_prime,
    int N,
    int C) {
    const int blockSize = blockDim.x;  // should be 256
    int total = N * C;
    // Each thread processes two consecutive elements
    int base = (blockIdx.x * blockSize + threadIdx.x) * 2;
    if (base >= total) return;

    // First element
    int idx0 = base;
    int ch0  = idx0 % C;
    float a0 = __ldg(alpha + ch0);
    float b0 = __ldg(beta_prime + ch0);
    float x0 = __bfloat162float(input[idx0]);
    float y0 = __fmaf_rn(a0, x0, b0);
    __nv_bfloat16 o0 = __float2bfloat16(y0);
    output[idx0] = o0;

    // Second element if in bounds
    int idx1 = base + 1;
    if (idx1 < total) {
        int ch1  = idx1 % C;
        float a1 = __ldg(alpha + ch1);
        float b1 = __ldg(beta_prime + ch1);
        float x1 = __bfloat162float(input[idx1]);
        float y1 = __fmaf_rn(a1, x1, b1);
        __nv_bfloat16 o1 = __float2bfloat16(y1);
        output[idx1] = o1;
    }
}

// External C wrapper for the optimized batch normalization operator
extern "C" void batch_norm_bf16_optimized(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    const float          epsilon,
    const int            N,
    const int            C) 
{
    // 1) allocate intermediate alpha and beta_prime
    float *d_alpha = nullptr, *d_beta_prime = nullptr;
    cudaMalloc(&d_alpha,      C * sizeof(float));
    cudaMalloc(&d_beta_prime, C * sizeof(float));

    // 2) launch fused-stats kernel
    const int threadsStats = 256;
    dim3      blockStats(threadsStats);
    dim3      gridStats (C);
    batch_norm_fused_stats_prepare_bf16_kernel_optimized<<<gridStats, blockStats>>>(
        input,
        gamma,
        beta,
        d_alpha,
        d_beta_prime,
        N,
        C,
        epsilon
    );
    cudaDeviceSynchronize();

    // 3) launch apply kernel (each thread does 2 elements)
    const int totalElems   = N * C;
    const int elemsPerTh   = 2;
    const int threadsApply = 256;
    int       pairCount    = (totalElems + elemsPerTh - 1) / elemsPerTh;
    int       blocksApply  = (pairCount + threadsApply - 1) / threadsApply;
    dim3      blockApply(threadsApply);
    dim3      gridApply (blocksApply);
    batch_norm_apply_bf16_kernel_optimized<<<gridApply, blockApply>>>(
        output,
        input,
        d_alpha,
        d_beta_prime,
        N,
        C
    );
    cudaDeviceSynchronize();

    // 4) clean up
    cudaFree(d_alpha);
    cudaFree(d_beta_prime);
}
