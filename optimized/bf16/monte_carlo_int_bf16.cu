#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>

// Warp-level reduction utility
template <unsigned int W>
__inline__ __device__ float warp_reduce_sum_Optimized(float val) {
    #pragma unroll
    for (int offset = W >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Monte Carlo kernel: each block computes a partial sum and writes to block_sums
__global__ void monte_carlo_int_bf16_kernel_optimized(
    float* block_sums,
    const __nv_bfloat16* __restrict__ y_samples,
    int N) {
    extern __shared__ float warp_sums[]; // one float per warp

    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Double-buffered prefetch + unrolled loop
    float sum = 0.0f;
    int idx = tid_global;
    float reg0 = 0.0f;
    if (idx < N) {
        reg0 = __bfloat162float(__ldg(y_samples + idx));
        idx += stride;
    }
    #pragma unroll 8
    while (idx < N) {
        float reg1 = __bfloat162float(__ldg(y_samples + idx));
        sum += reg0;
        reg0 = reg1;
        idx += stride;
    }
    sum += reg0;

    // Warp-level reduction
    unsigned int lane = threadIdx.x & (warpSize - 1);
    unsigned int wid = threadIdx.x / warpSize;
    float wsum = warp_reduce_sum_Optimized<32>(sum);
    if (lane == 0) {
        warp_sums[wid] = wsum;
    }
    __syncthreads();

    // Block-level reduction by first warp
    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        int numWarps = blockDim.x / warpSize;
        for (int i = 0; i < numWarps; ++i) {
            block_sum += warp_sums[i];
        }
        block_sums[blockIdx.x] = block_sum;
    }
}

// Reduction & finalize kernel: reduce block_sums and write final bfloat16 result
__global__ void reduce_block_sums_kernel_optimized(
    const float* block_sums,
    __nv_bfloat16* result,
    float a,
    float b,
    int N,
    int numBlocks) {
    extern __shared__ float warp_sums[]; // one float per warp

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load and partial reduce
    float sum = 0.0f;
    for (int i = tid; i < numBlocks; i += stride) {
        sum += block_sums[i];
    }

    // Warp-level reduction
    unsigned int lane = tid & (warpSize - 1);
    unsigned int wid = tid / warpSize;
    float wsum = warp_reduce_sum_Optimized<32>(sum);
    if (lane == 0) {
        warp_sums[wid] = wsum;
    }
    __syncthreads();

    // Final reduction by first warp
    float total = 0.0f;
    if (wid == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        if (lane < numWarps) {
            total = warp_sums[lane];
        }
        total = warp_reduce_sum_Optimized<32>(total);
    }
    __syncthreads();

    // Thread 0 writes result
    if (tid == 0) {
        float integral = ((b - a) / static_cast<float>(N)) * total;
        *result = __float2bfloat16_rn(integral);
    }
}

extern "C" void monte_carlo_int_bf16_optimized(
    const __nv_bfloat16* y_samples,
    __nv_bfloat16* result,
    float a,
    float b,
    int N) {
    // Determine grid and block sizes
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, 1024);

    // Allocate per-block sums
    float* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, blocks * sizeof(float));

    // Launch kernel1: compute per-block sums
    size_t sharedMem1 = (threadsPerBlock / 32) * sizeof(float);
    monte_carlo_int_bf16_kernel_optimized<<<blocks, threadsPerBlock, sharedMem1>>>(
        d_block_sums, y_samples, N);

    // Launch kernel2: reduce block_sums and finalize
    const int threads2 = 256;
    size_t sharedMem2 = (threads2 / 32) * sizeof(float);
    reduce_block_sums_kernel_optimized<<<1, threads2, sharedMem2>>>(
        d_block_sums, result, a, b, N, blocks);

    // Synchronize and cleanup
    cudaDeviceSynchronize();
    cudaFree(d_block_sums);
}