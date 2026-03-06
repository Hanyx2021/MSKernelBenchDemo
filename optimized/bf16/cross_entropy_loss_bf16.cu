#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Optimized kernel with block-level reduction, loop unrolling, and read-only cache loads
template <unsigned int BLOCK_SIZE>
__global__ void cross_entropy_loss_bf16_kernel_optimized(
    float* loss,
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ Y,
    int C) {
    extern __shared__ float sdata[];

    unsigned int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    const int UNROLL = 4;

    // Unrolled strided loop to increase ILP
    unsigned int i = tid_global;
    unsigned int stride_unroll = stride * UNROLL;
    for (; i + 3u * stride < (unsigned int)C; i += stride_unroll) {
        // Load and accumulate 4 elements
        float x0 = __bfloat162float(__ldg(&X[i]));
        float y0 = __bfloat162float(__ldg(&Y[i]));
        if (y0 != 0.0f) sum += -y0 * logf(fmaxf(x0, 1e-8f));

        unsigned int i1 = i + stride;
        float x1 = __bfloat162float(__ldg(&X[i1]));
        float y1 = __bfloat162float(__ldg(&Y[i1]));
        if (y1 != 0.0f) sum += -y1 * logf(fmaxf(x1, 1e-8f));

        unsigned int i2 = i + 2u * stride;
        float x2 = __bfloat162float(__ldg(&X[i2]));
        float y2 = __bfloat162float(__ldg(&Y[i2]));
        if (y2 != 0.0f) sum += -y2 * logf(fmaxf(x2, 1e-8f));

        unsigned int i3 = i + 3u * stride;
        float x3 = __bfloat162float(__ldg(&X[i3]));
        float y3 = __bfloat162float(__ldg(&Y[i3]));
        if (y3 != 0.0f) sum += -y3 * logf(fmaxf(x3, 1e-8f));
    }

    // Remainder loop
    for (; i < (unsigned int)C; i += stride) {
        float x_val = __bfloat162float(__ldg(&X[i]));
        float y_val = __bfloat162float(__ldg(&Y[i]));
        if (y_val != 0.0f) {
            sum += -y_val * logf(fmaxf(x_val, 1e-8f));
        }
    }

    // Block-level reduction
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // One atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

extern "C" void cross_entropy_loss_bf16_optimized(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int C) {
    const int threadsPerBlock = 256;
    int blocks = (C + threadsPerBlock - 1) / threadsPerBlock;
    blocks = (blocks > 1024 ? 1024 : blocks);
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch optimized kernel
    cross_entropy_loss_bf16_kernel_optimized<threadsPerBlock>
        <<<blocks, threadsPerBlock, sharedMemSize>>>(loss, X, Y, C);
    cudaDeviceSynchronize();
}