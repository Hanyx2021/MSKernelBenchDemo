#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <float.h>

#define THREADS_PER_BLOCK 256

// Optimized block-parallel softmax for bfloat16 using warp-shuffle reductions
__global__ void softmax_bf16_kernel_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int N,
    int C) {
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int lane   = tid % warpSize;
    const __nv_bfloat16* input_row = input + row * C;
    __nv_bfloat16* out_row = out + row * C;

    // 1. Compute local max over this thread's elements
    float local_max = -FLT_MAX;
    for (int j = tid; j < C; j += blockDim.x) {
        float v = __bfloat162float(input_row[j]);
        local_max = fmaxf(local_max, v);
    }
    
    // 2. Warp-level reduction to compute per-warp max
    unsigned int full_mask = 0xffffffff;
    float warp_max = local_max;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(full_mask, warp_max, offset);
        warp_max = fmaxf(warp_max, other);
    }

    // Shared memory for inter-warp reduction
    __shared__ float shared_max[32];
    if (lane == 0) {
        shared_max[warpId] = warp_max;
    }
    __syncthreads();

    // 3. First warp reduces across warp leaders to get global max
    float block_max = -FLT_MAX;
    if (warpId == 0) {
        float v = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(full_mask, v, offset);
            v = fmaxf(v, other);
        }
        if (lane == 0) {
            shared_max[0] = v;
        }
    }
    __syncthreads();
    float row_max = shared_max[0];

    // 4. Compute local sum of exp(val - row_max)
    float local_sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        float v = __bfloat162float(input_row[j]);
        float e = expf(v - row_max);
        local_sum += e;
    }

    // 5. Warp-level reduction to compute per-warp sums
    float warp_sum = local_sum;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(full_mask, warp_sum, offset);
        warp_sum += other;
    }

    __shared__ float shared_sum[32];
    if (lane == 0) {
        shared_sum[warpId] = warp_sum;
    }
    __syncthreads();

    // 6. First warp reduces across warp leaders to get global sum
    if (warpId == 0) {
        float v = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(full_mask, v, offset);
            v += other;
        }
        if (lane == 0) {
            shared_sum[0] = v;
        }
    }
    __syncthreads();
    float inv_sum = 1.0f / shared_sum[0];

    // 7. Write normalized output
    for (int j = tid; j < C; j += blockDim.x) {
        float v = __bfloat162float(input_row[j]);
        float e = expf(v - row_max) * inv_sum;
        out_row[j] = __float2bfloat16(e);
    }
}

extern "C" void softmax_bf16_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int N,
    int C) {
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid(N, 1, 1);
    softmax_bf16_kernel_optimized<<<grid, block>>>(out, input, N, C);
    cudaDeviceSynchronize();
}