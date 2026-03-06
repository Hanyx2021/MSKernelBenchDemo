#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32

// First stage: block-level partial sum reduction (no atomics)
__global__ void rms_bf16_partial_kernel_optimized(
    const __nv_bfloat16* input,
    float* d_partial,
    int N) {
    extern __shared__ float sdata[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    // accumulate v*v in register
    for (int i = tid; i < N; i += stride) {
        float v = __bfloat162float(input[i]);
        sum += v * v;
    }
    // intra-warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // warp-level sum in shared memory
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();
    // one warp reduces all warp sums
    if (warpId == 0) {
        int nWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float warpSum = (lane < nWarps) ? sdata[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }
        if (lane == 0) {
            d_partial[blockIdx.x] = warpSum;
        }
    }
}

// Second stage: single-block reduction over partial sums
__global__ void rms_bf16_finalize_kernel_optimized(
    const float* d_partial,
    float* d_rms,
    int numBlocks) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    // accumulate partial sums
    for (int i = tid; i < numBlocks; i += blockDim.x) {
        sum += d_partial[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    // tree-based reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *d_rms = sdata[0];
    }
}

// Normalization kernel unchanged except name
__global__ void rms_norm_bf16_kernel_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const float* d_rms,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float epsilon,
    int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float s = *d_rms;
    float rms = sqrtf(s / (float)N + epsilon);
    for (int i = tid; i < N; i += stride) {
        float x = __bfloat162float(input[i]);
        float x_hat = x / rms;
        float w = __bfloat162float(weight[i]);
        float b = __bfloat162float(bias[i]);
        output[i] = __float2bfloat16(w * x_hat + b);
    }
}

extern "C" void rms_norm_bf16_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const float epsilon,
    const int N) {
    // Allocate device memory for the final rms and block partial sums
    float* d_rms = nullptr;
    float* d_partial = nullptr;
    cudaMalloc(&d_rms, sizeof(float));
    // Determine block count and allocate partials
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    blocks = blocks > 1024 ? 1024 : blocks;
    cudaMalloc(&d_partial, sizeof(float) * blocks);
    cudaMemset(d_rms, 0, sizeof(float));

    // 1) Partial sum kernel
    int sharedMemSize = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);
    rms_bf16_partial_kernel_optimized<<<blocks, threads, sharedMemSize>>>(
        input, d_partial, N);

    // 2) Finalize sum kernel
    int threads2 = 1;
    while (threads2 < blocks) threads2 <<= 1;
    threads2 = threads2 > 1024 ? 1024 : threads2;
    int sharedMem2 = threads2 * sizeof(float);
    rms_bf16_finalize_kernel_optimized<<<1, threads2, sharedMem2>>>(
        d_partial, d_rms, blocks);

    // 3) Normalization kernel
    rms_norm_bf16_kernel_optimized<<<blocks, threads>>>(
        output, input, d_rms, weight, bias, epsilon, N);

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_rms);
    cudaFree(d_partial);
}
