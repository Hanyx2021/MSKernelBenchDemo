#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Optimized BF16 dot product kernel with higher unroll (8 elements) and software pipelining
static const int WARP_SIZE_OPTIM = 32;
static const int MAX_WARPS_OPTIM = 32;  // supports up to 1024 threads per block

// Union to unpack four bf16 values from a 64-bit packed word
union BF16x4Optimized {
    uint64_t packed;
    __nv_bfloat16 v[4];
};

__global__ void dot_product_bf16_kernel_optimized(
    float*            __restrict__ loss,
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ Y,
    const int             N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    // Check 8-byte alignment for vectorized bf16 loads
    bool aligned8 = (((uintptr_t)X & 7) == 0) && (((uintptr_t)Y & 7) == 0);

    if (aligned8) {
        const uint64_t* Xp64 = reinterpret_cast<const uint64_t*>(X);
        const uint64_t* Yp64 = reinterpret_cast<const uint64_t*>(Y);
        unsigned int groupCount = N >> 3;       // number of complete 8-element groups
        unsigned int stepG = stride;

        // Software pipelined prefetch for two groups
        unsigned int g0 = tid;
        unsigned int g1 = g0 + stepG;
        uint64_t x0_lo = 0, x0_hi = 0, y0_lo = 0, y0_hi = 0;
        uint64_t x1_lo = 0, x1_hi = 0, y1_lo = 0, y1_hi = 0;
        if (g0 < groupCount) {
            unsigned int base0 = g0 * 2;
            x0_lo = __ldg(Xp64 + base0);
            x0_hi = __ldg(Xp64 + base0 + 1);
            y0_lo = __ldg(Yp64 + base0);
            y0_hi = __ldg(Yp64 + base0 + 1);
        }
        if (g1 < groupCount) {
            unsigned int base1 = g1 * 2;
            x1_lo = __ldg(Xp64 + base1);
            x1_hi = __ldg(Xp64 + base1 + 1);
            y1_lo = __ldg(Yp64 + base1);
            y1_hi = __ldg(Yp64 + base1 + 1);
        }

        // Main pipelined loop
        while (g0 < groupCount) {
            // Convert & multiply group g0
            BF16x4Optimized ux_lo{ x0_lo };
            BF16x4Optimized ux_hi{ x0_hi };
            BF16x4Optimized uy_lo{ y0_lo };
            BF16x4Optimized uy_hi{ y0_hi };
            float fx0 = __bfloat162float(ux_lo.v[0]);
            float fx1 = __bfloat162float(ux_lo.v[1]);
            float fx2 = __bfloat162float(ux_lo.v[2]);
            float fx3 = __bfloat162float(ux_lo.v[3]);
            float fx4 = __bfloat162float(ux_hi.v[0]);
            float fx5 = __bfloat162float(ux_hi.v[1]);
            float fx6 = __bfloat162float(ux_hi.v[2]);
            float fx7 = __bfloat162float(ux_hi.v[3]);
            float fy0 = __bfloat162float(uy_lo.v[0]);
            float fy1 = __bfloat162float(uy_lo.v[1]);
            float fy2 = __bfloat162float(uy_lo.v[2]);
            float fy3 = __bfloat162float(uy_lo.v[3]);
            float fy4 = __bfloat162float(uy_hi.v[0]);
            float fy5 = __bfloat162float(uy_hi.v[1]);
            float fy6 = __bfloat162float(uy_hi.v[2]);
            float fy7 = __bfloat162float(uy_hi.v[3]);
            sum += fx0*fy0 + fx1*fy1 + fx2*fy2 + fx3*fy3
                 + fx4*fy4 + fx5*fy5 + fx6*fy6 + fx7*fy7;

            // Advance pipeline
            g0 = g1;
            x0_lo = x1_lo; x0_hi = x1_hi; y0_lo = y1_lo; y0_hi = y1_hi;
            g1 += stepG;
            if (g1 < groupCount) {
                unsigned int base1n = g1 * 2;
                x1_lo = __ldg(Xp64 + base1n);
                x1_hi = __ldg(Xp64 + base1n + 1);
                y1_lo = __ldg(Yp64 + base1n);
                y1_hi = __ldg(Yp64 + base1n + 1);
            }
        }

        // Cleanup leftover elements (<8 per thread) using scalar loads
        unsigned int processed = groupCount * 8;
        for (unsigned int j = processed + tid; j < (unsigned int)N; j += stride) {
            float xf = __bfloat162float(__ldg(X + j));
            float yf = __bfloat162float(__ldg(Y + j));
            sum += xf * yf;
        }
    } else {
        // Fallback to scalar/vector loads (unaligned)
        for (unsigned int i = tid * 2; i + 1 < (unsigned int)N; i += stride * 2) {
            float x0 = __bfloat162float(__ldg(X + i));
            float x1 = __bfloat162float(__ldg(X + i + 1));
            float y0 = __bfloat162float(__ldg(Y + i));
            float y1 = __bfloat162float(__ldg(Y + i + 1));
            sum += x0 * y0 + x1 * y1;
        }
        if ((N & 1) && tid == (unsigned int)(N - 1)) {
            float xt = __bfloat162float(__ldg(X + tid));
            float yt = __bfloat162float(__ldg(Y + tid));
            sum += xt * yt;
        }
    }

    // Intra-warp reduction using shuffle down
    unsigned int lane   = threadIdx.x & (WARP_SIZE_OPTIM - 1);
    unsigned int warpId = threadIdx.x >> 5;
    unsigned int fullMask = 0xffffffffu;
    for (int offset = WARP_SIZE_OPTIM / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(fullMask, sum, offset);
    }

    // Shared memory for warp sums
    __shared__ float warp_sums[MAX_WARPS_OPTIM];
    if (lane == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    float block_sum = 0.0f;
    if (threadIdx.x < WARP_SIZE_OPTIM) {
        block_sum = warp_sums[lane];
    }
    for (int offset = WARP_SIZE_OPTIM / 2; offset > 0; offset >>= 1) {
        block_sum += __shfl_down_sync(fullMask, block_sum, offset);
    }

    // Single atomic update per block
    if (lane == 0) {
        atomicAdd(loss, block_sum);
    }
}

extern "C" void dot_product_bf16_optimized(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {
    // Tuned block size for occupancy under higher unroll/register pressure
    const int threadsPerBlock = 256;
    int groupCount = N >> 3;
    int blocks = (groupCount + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;

    dot_product_bf16_kernel_optimized<<<blocks, threadsPerBlock>>>(loss, X, Y, N);
    cudaDeviceSynchronize();
}
