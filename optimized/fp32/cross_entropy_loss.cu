#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Optimized cross entropy loss kernel with loop unrolling, read-only cache, and block-level reduction
__global__ void cross_entropy_loss_kernel_optimized(
    float* loss,
    const float* __restrict__ X,
    const float* __restrict__ Y,
    int C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Unrolled accumulators
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    const float eps = 1e-8f;
    int i = tid;
    int limit = C - 3 * stride;

    // Main unrolled loop (factor 4)
    for (; i < limit; i += 4 * stride) {
        float y0 = __ldg(&Y[i]);
        float x0 = fmaxf(__ldg(&X[i]), eps);
        float y1 = __ldg(&Y[i + stride]);
        float x1 = fmaxf(__ldg(&X[i + stride]), eps);
        float y2 = __ldg(&Y[i + 2 * stride]);
        float x2 = fmaxf(__ldg(&X[i + 2 * stride]), eps);
        float y3 = __ldg(&Y[i + 3 * stride]);
        float x3 = fmaxf(__ldg(&X[i + 3 * stride]), eps);
        if (y0 != 0.0f) sum0 += -y0 * logf(x0);
        if (y1 != 0.0f) sum1 += -y1 * logf(x1);
        if (y2 != 0.0f) sum2 += -y2 * logf(x2);
        if (y3 != 0.0f) sum3 += -y3 * logf(x3);
    }

    // Epilogue for remaining elements
    for (; i < C; i += stride) {
        float y = __ldg(&Y[i]);
        if (y != 0.0f) {
            float x = fmaxf(__ldg(&X[i]), eps);
            sum0 += -y * logf(x);
        }
    }

    // Combine accumulators
    float sum = sum0 + sum1 + sum2 + sum3;

    // Warp-level reduction
    unsigned int lane = threadIdx.x & 31;
    unsigned int warpId = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Shared memory for per-warp sums
    __shared__ float warp_sums[32];
    if (lane == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warpId == 0) {
        sum = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            atomicAdd(loss, sum);
        }
    }
}

extern "C" void cross_entropy_loss_optimized(
    float* loss,
    const float* X,
    const float* Y,
    int C) {
    const int threadsPerBlock = 256;
    int blocks = (C + threadsPerBlock - 1) / threadsPerBlock;
    blocks = (blocks > 1024) ? 1024 : blocks;

    cross_entropy_loss_kernel_optimized<<<blocks, threadsPerBlock>>>(
        loss, X, Y, C);
    cudaDeviceSynchronize();
}
