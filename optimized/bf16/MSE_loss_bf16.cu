#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Configuration for optimized kernels
struct MSE_loss_bf16_OptimizedConfig {
    static constexpr int threadsPerBlock = 256;
    static constexpr int maxBlocks = 1024;
};

// Stage 1: Per-block partial sum computation and reduction
__global__ void MSE_loss_bf16_stage1_kernel_optimized(
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    int N,
    float* partial_sums) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    unsigned int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    // Compute strided local sum
    for (int i = idx; i < N; i += stride) {
        float x = __bfloat162float(X[i]);
        float y = __bfloat162float(Y[i]);
        float d = x - y;
        local_sum += d * d;
    }
    // Store in shared memory
    sdata[tid] = local_sum;
    __syncthreads();
    // In-block reduction
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Stage 2: Reduce partial sums and perform final division
__global__ void MSE_loss_bf16_stage2_kernel_optimized(
    float* loss,
    const float* partial_sums,
    int numBlocks,
    int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    // Load partial sums
    if (tid < numBlocks) {
        sdata[tid] = partial_sums[tid];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    // Reduction
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    // Final division and write out
    if (tid == 0) {
        loss[0] = sdata[0] / static_cast<float>(N);
    }
}

// External C wrapper for optimized operator
extern "C" void MSE_loss_bf16_optimized(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {
    // Configure stage 1
    int threads1 = MSE_loss_bf16_OptimizedConfig::threadsPerBlock;
    int blocks = (N + threads1 - 1) / threads1;
    blocks = (blocks > MSE_loss_bf16_OptimizedConfig::maxBlocks)
             ? MSE_loss_bf16_OptimizedConfig::maxBlocks
             : blocks;
    // Allocate temporary partial sums
    float* partial_sums = nullptr;
    cudaMalloc(&partial_sums, blocks * sizeof(float));
    // Launch stage 1
    size_t sharedMem1 = threads1 * sizeof(float);
    MSE_loss_bf16_stage1_kernel_optimized<<<blocks, threads1, sharedMem1>>>(
        X, Y, N, partial_sums);
    // Configure stage 2
    int threads2 = 1;
    while (threads2 < blocks) threads2 <<= 1;
    if (threads2 > MSE_loss_bf16_OptimizedConfig::maxBlocks) {
        threads2 = MSE_loss_bf16_OptimizedConfig::maxBlocks;
    }
    size_t sharedMem2 = threads2 * sizeof(float);
    // Launch stage 2 (reduction + division)
    MSE_loss_bf16_stage2_kernel_optimized<<<1, threads2, sharedMem2>>>(
        loss, partial_sums, blocks, N);
    // Clean up
    cudaFree(partial_sums);
    cudaDeviceSynchronize();
}