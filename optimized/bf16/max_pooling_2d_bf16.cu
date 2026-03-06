#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>

// Block dimensions for all specializations
#define BLOCK_W 16
#define BLOCK_H 16

extern "C"
__global__ void max_pool2d_bf16_k2s1_kernel_optimized(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int padding) {
    const int k = 2;
    const int s = 1;
    const int tileH = (BLOCK_H - 1) * s + k;  // 17
    const int tileW = (BLOCK_W - 1) * s + k;  // 17

    // We split the tile into two stripes in H
    const int stripes = 2;
    const int stripeH = (tileH + stripes - 1) / stripes; // 9

    extern __shared__ float shared_mem[];
    float* sdata0 = shared_mem;
    float* sdata1 = shared_mem + stripeH * tileW;

    const int numThreads = BLOCK_W * BLOCK_H;  // 256
    int z    = blockIdx.z;
    int n    = z / C;
    int c    = z % C;
    int h0   = blockIdx.y * BLOCK_H * s - padding;
    int w0   = blockIdx.x * BLOCK_W * s - padding;
    int tid  = threadIdx.y * BLOCK_W + threadIdx.x;

    // ------- stripe 0 -------
    int tileOffset   = 0;
    int thisStripeH  = min(stripeH, tileH - tileOffset);
    int stripeSize   = thisStripeH * tileW;
    int ITER0        = (stripeSize + numThreads - 1) / numThreads;
#pragma unroll
    for (int i = 0; i < ITER0; ++i) {
        int idx = tid + i * numThreads;
        if (idx < stripeSize) {
            int tr   = idx / tileW;
            int tc   = idx % tileW;
            int h_in = h0 + tr;
            int w_in = w0 + tc;
            float val = -FLT_MAX;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                val = __bfloat162float(input[in_idx]);
            }
            sdata0[tr * tileW + tc] = val;
        }
    }

    // ------- stripe 1 -------
    tileOffset      = stripeH;
    thisStripeH     = tileH - tileOffset;
    stripeSize      = thisStripeH * tileW;
    int ITER1       = (stripeSize + numThreads - 1) / numThreads;
#pragma unroll
    for (int i = 0; i < ITER1; ++i) {
        int idx = tid + i * numThreads;
        if (idx < stripeSize) {
            int tr   = idx / tileW;
            int tc   = idx % tileW;
            int h_in = h0 + tileOffset + tr;
            int w_in = w0 + tc;
            float val = -FLT_MAX;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                val = __bfloat162float(input[in_idx]);
            }
            sdata1[tr * tileW + tc] = val;
        }
    }

    // All data is now in shared_mem
    __syncthreads();

    int h_out = blockIdx.y * BLOCK_H + threadIdx.y;
    int w_out = blockIdx.x * BLOCK_W + threadIdx.x;
    if (h_out < out_H && w_out < out_W) {
        int row_off = threadIdx.y * s;
        int col_off = threadIdx.x * s;
        // 2×2 max
        float v0 = shared_mem[(row_off + 0) * tileW + (col_off + 0)];
        float v1 = shared_mem[(row_off + 0) * tileW + (col_off + 1)];
        float v2 = shared_mem[(row_off + 1) * tileW + (col_off + 0)];
        float v3 = shared_mem[(row_off + 1) * tileW + (col_off + 1)];
        float m01 = v0 > v1 ? v0 : v1;
        float m23 = v2 > v3 ? v2 : v3;
        float max_val = m01 > m23 ? m01 : m23;
        int out_idx = ((n * C + c) * out_H + h_out) * out_W + w_out;
        output[out_idx] = __float2bfloat16(max_val);
    }
}

extern "C"
__global__ void max_pool2d_bf16_k3s1_kernel_optimized(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int padding) {
    const int k = 3;
    const int s = 1;
    const int tileH = (BLOCK_H - 1) * s + k;  // 18
    const int tileW = (BLOCK_W - 1) * s + k;  // 18
    const int tileSize = tileH * tileW;       // 324
    const int numThreads = BLOCK_W * BLOCK_H;  // 256

    int z = blockIdx.z;
    int n = z / C;
    int c = z % C;
    int h0 = blockIdx.y * BLOCK_H * s - padding;
    int w0 = blockIdx.x * BLOCK_W * s - padding;

    extern __shared__ float sdata[];
    int tid = threadIdx.y * BLOCK_W + threadIdx.x;
    const int LOAD_ITERS = (tileSize + numThreads - 1) / numThreads; // 2
#pragma unroll
    for (int i = 0; i < LOAD_ITERS; ++i) {
        int idx = tid + i * numThreads;
        if (idx < tileSize) {
            int tr = idx / tileW;
            int tc = idx % tileW;
            int h_in = h0 + tr;
            int w_in = w0 + tc;
            float val = -FLT_MAX;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                val = __bfloat162float(input[in_idx]);
            }
            sdata[idx] = val;
        }
    }
    __syncthreads();

    int h_out = blockIdx.y * BLOCK_H + threadIdx.y;
    int w_out = blockIdx.x * BLOCK_W + threadIdx.x;
    if (h_out < out_H && w_out < out_W) {
        int row_off = threadIdx.y * s;
        int col_off = threadIdx.x * s;
        // Unrolled 3x3 max
        float m = -FLT_MAX;
#pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                float v = sdata[(row_off + dy) * tileW + (col_off + dx)];
                m = v > m ? v : m;
            }
        }
        int out_idx = ((n * C + c) * out_H + h_out) * out_W + w_out;
        output[out_idx] = __float2bfloat16(m);
    }
}

extern "C"
__global__ void max_pool2d_bf16_k4s2_kernel_optimized(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int padding) {
    const int k = 4;
    const int s = 2;
    const int tileH = (BLOCK_H - 1) * s + k;  // 34
    const int tileW = (BLOCK_W - 1) * s + k;  // 34

    const int stripes  = 2;
    const int stripeH  = (tileH + stripes - 1) / stripes; // 17

    extern __shared__ float shared_mem2[];
    float* sdata0 = shared_mem2;
    float* sdata1 = shared_mem2 + stripeH * tileW;

    const int numThreads = BLOCK_W * BLOCK_H;  // 256
    int z = blockIdx.z;
    int n = z / C;
    int c = z % C;
    int h0 = blockIdx.y * BLOCK_H * s - padding;
    int w0 = blockIdx.x * BLOCK_W * s - padding;
    int tid = threadIdx.y * BLOCK_W + threadIdx.x;

    // stripe 0
    int tileOffset    = 0;
    int thisStripeH   = min(stripeH, tileH - tileOffset);
    int stripeSize    = thisStripeH * tileW;
    int ITER0         = (stripeSize + numThreads - 1) / numThreads;
#pragma unroll
    for (int i = 0; i < ITER0; ++i) {
        int idx = tid + i * numThreads;
        if (idx < stripeSize) {
            int tr   = idx / tileW;
            int tc   = idx % tileW;
            int h_in = h0 + tr;
            int w_in = w0 + tc;
            float val = -FLT_MAX;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                val = __bfloat162float(input[in_idx]);
            }
            sdata0[tr * tileW + tc] = val;
        }
    }

    // stripe 1
    tileOffset    = stripeH;
    thisStripeH   = tileH - tileOffset;
    stripeSize    = thisStripeH * tileW;
    int ITER1     = (stripeSize + numThreads - 1) / numThreads;
#pragma unroll
    for (int i = 0; i < ITER1; ++i) {
        int idx = tid + i * numThreads;
        if (idx < stripeSize) {
            int tr   = idx / tileW;
            int tc   = idx % tileW;
            int h_in = h0 + tileOffset + tr;
            int w_in = w0 + tc;
            float val = -FLT_MAX;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int in_idx = ((n * C + c) * H + h_in) * W + w_in;
                val = __bfloat162float(input[in_idx]);
            }
            sdata1[tr * tileW + tc] = val;
        }
    }

    __syncthreads();

    int h_out = blockIdx.y * BLOCK_H + threadIdx.y;
    int w_out = blockIdx.x * BLOCK_W + threadIdx.x;
    if (h_out < out_H && w_out < out_W) {
        int row_off = threadIdx.y * s;
        int col_off = threadIdx.x * s;
        float m = -FLT_MAX;
#pragma unroll
        for (int dy = 0; dy < 4; ++dy) {
#pragma unroll
            for (int dx = 0; dx < 4; ++dx) {
                float v = shared_mem2[(row_off + dy) * tileW + (col_off + dx)];
                m = v > m ? v : m;
            }
        }
        int out_idx = ((n * C + c) * out_H + h_out) * out_W + w_out;
        output[out_idx] = __float2bfloat16(m);
    }
}

extern "C" void max_pooling_2d_bf16_optimized(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int N,
    int C,
    int H,
    int W,
    int kernel_size,
    int stride,
    int padding) {
    int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    int out_W = (W + 2 * padding - kernel_size) / stride + 1;
    if (out_H <= 0 || out_W <= 0) return;

    dim3 blockSize(BLOCK_W, BLOCK_H);
    dim3 gridSize(
        (out_W + BLOCK_W - 1) / BLOCK_W,
        (out_H + BLOCK_H - 1) / BLOCK_H,
        N * C);

    if (kernel_size == 2 && stride == 1) {
        size_t sharedMemBytes = ((BLOCK_H - 1) * 1 + 2) * ((BLOCK_W - 1) * 1 + 2) * sizeof(float);
        max_pool2d_bf16_k2s1_kernel_optimized<<<gridSize, blockSize, sharedMemBytes>>>(
            input, output, N, C, H, W, out_H, out_W, padding);
    } else if (kernel_size == 3 && stride == 1) {
        size_t sharedMemBytes = ((BLOCK_H - 1) * 1 + 3) * ((BLOCK_W - 1) * 1 + 3) * sizeof(float);
        max_pool2d_bf16_k3s1_kernel_optimized<<<gridSize, blockSize, sharedMemBytes>>>(
            input, output, N, C, H, W, out_H, out_W, padding);
    } else if (kernel_size == 4 && stride == 2) {
        size_t sharedMemBytes = ((BLOCK_H - 1) * 2 + 4) * ((BLOCK_W - 1) * 2 + 4) * sizeof(float);
        max_pool2d_bf16_k4s2_kernel_optimized<<<gridSize, blockSize, sharedMemBytes>>>(
            input, output, N, C, H, W, out_H, out_W, padding);
    } else {
        // Fallback: unsupported specialization
        return;
    }
    cudaDeviceSynchronize();
}
