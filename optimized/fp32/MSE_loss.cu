#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Configuration struct for optimized kernel parameters
struct MSELossOptimizedConfig {
    static constexpr int THREADS_PER_BLOCK = 256;
    static constexpr int MAX_BLOCKS = 1024;
    static constexpr int THREADS_FINAL = 1024;
    static constexpr int N_THRESHOLD = 16384;  // Threshold for single-stage kernel
};

// First stage: block-wise reduction with vectorized loads, loop unrolling, and warp-shuffle
__global__ void MSE_loss_stage1_kernel_optimized(
    float* partial_sums,
    const float* X,
    const float* Y,
    int N) {
    extern __shared__ float warp_sums[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    const int WARP_SIZE = 32;

    int N4 = N / 4;
    const float4* X4 = reinterpret_cast<const float4*>(X);
    const float4* Y4 = reinterpret_cast<const float4*>(Y);

    float sum = 0.0f;
    int i4 = tid;
    // Unroll by 4 vector-iterations
    for (; i4 + 3 * total_threads < N4; i4 += total_threads * 4) {
        float4 x0 = X4[i4];           float4 y0 = Y4[i4];
        float4 x1 = X4[i4 + total_threads];     float4 y1 = Y4[i4 + total_threads];
        float4 x2 = X4[i4 + 2 * total_threads]; float4 y2 = Y4[i4 + 2 * total_threads];
        float4 x3 = X4[i4 + 3 * total_threads]; float4 y3 = Y4[i4 + 3 * total_threads];
        sum += (x0.x - y0.x) * (x0.x - y0.x);
        sum += (x0.y - y0.y) * (x0.y - y0.y);
        sum += (x0.z - y0.z) * (x0.z - y0.z);
        sum += (x0.w - y0.w) * (x0.w - y0.w);
        sum += (x1.x - y1.x) * (x1.x - y1.x);
        sum += (x1.y - y1.y) * (x1.y - y1.y);
        sum += (x1.z - y1.z) * (x1.z - y1.z);
        sum += (x1.w - y1.w) * (x1.w - y1.w);
        sum += (x2.x - y2.x) * (x2.x - y2.x);
        sum += (x2.y - y2.y) * (x2.y - y2.y);
        sum += (x2.z - y2.z) * (x2.z - y2.z);
        sum += (x2.w - y2.w) * (x2.w - y2.w);
        sum += (x3.x - y3.x) * (x3.x - y3.x);
        sum += (x3.y - y3.y) * (x3.y - y3.y);
        sum += (x3.z - y3.z) * (x3.z - y3.z);
        sum += (x3.w - y3.w) * (x3.w - y3.w);
    }
    // Remaining vectorized iterations
    for (; i4 < N4; i4 += total_threads) {
        float4 x = X4[i4];
        float4 y = Y4[i4];
        sum += (x.x - y.x) * (x.x - y.x);
        sum += (x.y - y.y) * (x.y - y.y);
        sum += (x.z - y.z) * (x.z - y.z);
        sum += (x.w - y.w) * (x.w - y.w);
    }
    // Handle tail elements
    int tail_start = N4 * 4;
    for (int i = tail_start + tid; i < N; i += total_threads) {
        float diff = X[i] - Y[i];
        sum += diff * diff;
    }

    // In-warp reduction using shuffle
    unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    int lane   = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // Let first warp finish the reduction of per-warp sums
    if (warpId == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        float wsum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            wsum += __shfl_down_sync(mask, wsum, offset);
        }
        if (lane == 0) {
            partial_sums[blockIdx.x] = wsum;
        }
    }
}

// Second stage: final reduction of partial sums and divide by N
__global__ void MSE_loss_final_reduce_kernel_optimized(
    float* loss,
    const float* partial_sums,
    int num_blocks,
    int N) {
    extern __shared__ float sdata[];
    int tx = threadIdx.x;

    float sum = 0.0f;
    for (int i = tx; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    sdata[tx] = sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tx < offset) {
            sdata[tx] += sdata[tx + offset];
        }
        __syncthreads();
    }

    if (tx == 0) {
        *loss = sdata[0] / static_cast<float>(N);
    }
}

// Fused single-block kernel for small N
__global__ void MSE_loss_fused_kernel_optimized(
    float* loss,
    const float* X,
    const float* Y,
    int N) {
    extern __shared__ float warp_sums[];

    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    const int WARP_SIZE = 32;

    int N4 = N / 4;
    const float4* X4 = reinterpret_cast<const float4*>(X);
    const float4* Y4 = reinterpret_cast<const float4*>(Y);

    float sum = 0.0f;
    int i4 = tid;
    // Unroll by 4 vector-iterations
    for (; i4 + 3 * total_threads < N4; i4 += total_threads * 4) {
        float4 x0 = X4[i4];           float4 y0 = Y4[i4];
        float4 x1 = X4[i4 + total_threads];     float4 y1 = Y4[i4 + total_threads];
        float4 x2 = X4[i4 + 2 * total_threads]; float4 y2 = Y4[i4 + 2 * total_threads];
        float4 x3 = X4[i4 + 3 * total_threads]; float4 y3 = Y4[i4 + 3 * total_threads];
        sum += (x0.x - y0.x) * (x0.x - y0.x);
        sum += (x0.y - y0.y) * (x0.y - y0.y);
        sum += (x0.z - y0.z) * (x0.z - y0.z);
        sum += (x0.w - y0.w) * (x0.w - y0.w);
        sum += (x1.x - y1.x) * (x1.x - y1.x);
        sum += (x1.y - y1.y) * (x1.y - y1.y);
        sum += (x1.z - y1.z) * (x1.z - y1.z);
        sum += (x1.w - y1.w) * (x1.w - y1.w);
        sum += (x2.x - y2.x) * (x2.x - y2.x);
        sum += (x2.y - y2.y) * (x2.y - y2.y);
        sum += (x2.z - y2.z) * (x2.z - y2.z);
        sum += (x2.w - y2.w) * (x2.w - y2.w);
        sum += (x3.x - y3.x) * (x3.x - y3.x);
        sum += (x3.y - y3.y) * (x3.y - y3.y);
        sum += (x3.z - y3.z) * (x3.z - y3.z);
        sum += (x3.w - y3.w) * (x3.w - y3.w);
    }
    // Remaining vectorized iterations
    for (; i4 < N4; i4 += total_threads) {
        float4 x = X4[i4];
        float4 y = Y4[i4];
        sum += (x.x - y.x) * (x.x - y.x);
        sum += (x.y - y.y) * (x.y - y.y);
        sum += (x.z - y.z) * (x.z - y.z);
        sum += (x.w - y.w) * (x.w - y.w);
    }
    // Handle tail elements
    int tail_start = N4 * 4;
    for (int i = tail_start + tid; i < N; i += total_threads) {
        float diff = X[i] - Y[i];
        sum += diff * diff;
    }

    // In-warp reduction using shuffle
    unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    int lane   = tid & (WARP_SIZE - 1);
    int warpId = tid >> 5;
    if (lane == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // Final block-level reduction in first warp
    if (warpId == 0) {
        int num_warps = total_threads / WARP_SIZE;
        float wsum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            wsum += __shfl_down_sync(mask, wsum, offset);
        }
        if (lane == 0) {
            *loss = wsum / static_cast<float>(N);
        }
    }
}

// External C wrapper for the optimized MSE loss computation
extern "C" void MSE_loss_optimized(
    float* loss,
    const float* X,
    const float* Y,
    int N) {
    if (N <= MSELossOptimizedConfig::N_THRESHOLD) {
        int threads = MSELossOptimizedConfig::THREADS_PER_BLOCK;
        if (threads > N) threads = N;
        int blocks = 1;
        int warps = (threads + 31) / 32;
        size_t shmem = warps * sizeof(float);
        MSE_loss_fused_kernel_optimized<<<blocks, threads, shmem>>>(loss, X, Y, N);
    } else {
        int blocks = (N + MSELossOptimizedConfig::THREADS_PER_BLOCK - 1)
                     / MSELossOptimizedConfig::THREADS_PER_BLOCK;
        if (blocks > MSELossOptimizedConfig::MAX_BLOCKS) {
            blocks = MSELossOptimizedConfig::MAX_BLOCKS;
        }
        float* d_partial_sums = nullptr;
        cudaMalloc(&d_partial_sums, blocks * sizeof(float));

        int threads = MSELossOptimizedConfig::THREADS_PER_BLOCK;
        constexpr int WARP_SIZE = 32;
        int warps_per_block = threads / WARP_SIZE;
        size_t shmem1 = warps_per_block * sizeof(float);
        MSE_loss_stage1_kernel_optimized<<<blocks, threads, shmem1>>>(
            d_partial_sums, X, Y, N);

        int threads_final = MSELossOptimizedConfig::THREADS_FINAL;
        if (threads_final > blocks) {
            threads_final = blocks;
        }
        size_t shmem2 = threads_final * sizeof(float);
        MSE_loss_final_reduce_kernel_optimized<<<1, threads_final, shmem2>>>(
            loss, d_partial_sums, blocks, N);
        cudaFree(d_partial_sums);
    }
    cudaDeviceSynchronize();
}
