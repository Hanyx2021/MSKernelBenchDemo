#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Optimized softmax kernel using warp-shuffle reductions
__global__ void softmax_kernel_optimized(
    float* out,
    const float* input,
    int N,
    int C) {
    extern __shared__ float smem[];  // shared memory for warp results

    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int numWarps = (numThreads + 31) / 32;
    int warpId = tid / 32;
    int lane = tid % 32;

    const float* input_row = input + row * C;
    float* out_row = out + row * C;

    // Phase 1: compute local max
    float local_max = -FLT_MAX;
    for (int j = tid; j < C; j += numThreads) {
        float v = input_row[j];
        if (v > local_max) local_max = v;
    }

    // Warp-level max reduction
    unsigned int mask = 0xffffffffu;
    float warp_max = local_max;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, warp_max, offset);
        warp_max = fmaxf(warp_max, other);
    }

    // Each warp writes its max to shared memory
    if (lane == 0) {
        smem[warpId] = warp_max;
    }
    __syncthreads();

    // Block-level max reduction by warp 0
    float max_val;
    if (warpId == 0) {
        float w = (tid < numWarps) ? smem[tid] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(mask, w, offset);
            w = fmaxf(w, other);
        }
        if (lane == 0) {
            smem[0] = w;
        }
    }
    __syncthreads();
    max_val = smem[0];

    // Phase 2: compute exponentials and local sum
    float local_sum = 0.0f;
    for (int j = tid; j < C; j += numThreads) {
        float tmp = expf(input_row[j] - max_val);
        out_row[j] = tmp;
        local_sum += tmp;
    }

    // Warp-level sum reduction
    float warp_sum = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, warp_sum, offset);
        warp_sum += other;
    }
    if (lane == 0) {
        smem[warpId] = warp_sum;
    }
    __syncthreads();

    // Block-level sum reduction by warp 0
    float sum_val;
    if (warpId == 0) {
        float w = (tid < numWarps) ? smem[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(mask, w, offset);
            w += other;
        }
        if (lane == 0) {
            smem[0] = w;
        }
    }
    __syncthreads();
    sum_val = smem[0];

    // Phase 3: normalize
    for (int j = tid; j < C; j += numThreads) {
        out_row[j] /= sum_val;
    }
}

// External wrapper for the optimized softmax kernel
extern "C" void softmax_optimized(
    float* out,
    const float* input,
    int N,
    int C) {
    // Determine thread/block configuration
    int threads = (C < 1024) ? C : 1024;
    int warps = (threads + 31) / 32;
    size_t shared_bytes = sizeof(float) * warps;
    dim3 block(threads, 1, 1);
    dim3 grid(N, 1, 1);

    softmax_kernel_optimized<<<grid, block, shared_bytes>>>(out, input, N, C);
    cudaDeviceSynchronize();
}