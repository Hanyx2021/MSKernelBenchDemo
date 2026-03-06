#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel: vectorized & read-only-cache loads with full-grid coverage
__global__ void dot_product_kernel_optimized(
    float* loss,
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const int N) {
    extern __shared__ float warp_sums[];

    // Number of float4 elements and remaining tail elements
    const int vecN = N >> 2;
    const int tail = N & 3;

    // Global thread index and grid-stride
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Cast to float4 pointers for vectorized loads
    const float4* X4 = reinterpret_cast<const float4*>(X);
    const float4* Y4 = reinterpret_cast<const float4*>(Y);

    float sum = 0.0f;
    // Vectorized grid-stride loop over float4 elements
    for (int i = tid; i < vecN; i += stride) {
        float4 x4 = __ldg(&X4[i]);
        float4 y4 = __ldg(&Y4[i]);
        sum += x4.x * y4.x + x4.y * y4.y + x4.z * y4.z + x4.w * y4.w;
    }
    // Handle tail scalar elements
    int offset = vecN << 2;
    for (int i = tid; i < tail; i += stride) {
        sum += __ldg(&X[offset + i]) * __ldg(&Y[offset + i]);
    }

    // Hierarchical warp-shuffle reduction
    unsigned int lane = threadIdx.x & 31;
    unsigned int warpId = threadIdx.x >> 5;
    unsigned int fullMask = 0xffffffff;

    // In-warp reduction
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(fullMask, sum, off);
    }
    // Write per-warp result to shared memory
    if (lane == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warpId == 0) {
        sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(fullMask, sum, off);
        }
        if (lane == 0) {
            atomicAdd(loss, sum);
        }
    }
}

// External C wrapper: dot_product optimized version
extern "C" void dot_product_optimized(
    float* loss,
    const float* X,
    const float* Y,
    const int N) {
    const int threadsPerBlock = 256;
    const int vecN = N >> 2;
    // Compute blocks to cover all float4 elements (no clamp)
    int blocks = (vecN + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks == 0) blocks = 1;

    // Shared memory: one float per warp
    size_t sharedMemBytes = (threadsPerBlock / 32) * sizeof(float);

    // Launch optimized kernel
    dot_product_kernel_optimized<<<blocks, threadsPerBlock, sharedMemBytes>>>(
        loss, X, Y, N);
    cudaDeviceSynchronize();
}