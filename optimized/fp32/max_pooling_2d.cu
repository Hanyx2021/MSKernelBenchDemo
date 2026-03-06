#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <algorithm>

// Optimized max pooling kernel with shared-memory tiling and halo
extern "C"
__global__ void max_pooling_2d_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    int out_H,
    int out_W,
    int kernel_size,
    int stride,
    int padding) {
    // compute batch and channel
    int z = blockIdx.z;
    int n = z / C;
    int c = z % C;

    // output coordinates
    int th = threadIdx.y;
    int tw = threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + th;
    int w_out = blockIdx.x * blockDim.x + tw;

    // compute tile dimensions including halo
    int tile_h = blockDim.y * stride + (kernel_size - 1);
    int tile_w = blockDim.x * stride + (kernel_size - 1);

    extern __shared__ float smem[];
    float* patch = smem;

    // compute starting global indices for this tile
    int global_h_start = (blockIdx.y * blockDim.y) * stride - padding;
    int global_w_start = (blockIdx.x * blockDim.x) * stride - padding;

    int numThreads = blockDim.y * blockDim.x;
    int tid = th * blockDim.x + tw;
    int total_elems = tile_h * tile_w;

    // cooperative load into shared memory with halo, padding filled with -FLT_MAX
    for (int idx = tid; idx < total_elems; idx += numThreads) {
        int y = idx / tile_w;
        int x = idx % tile_w;
        int gh = global_h_start + y;
        int gw = global_w_start + x;
        float val = -FLT_MAX;
        if (gh >= 0 && gh < H && gw >= 0 && gw < W) {
            int in_idx = ((n * C + c) * H + gh) * W + gw;
            val = input[in_idx];
        }
        patch[y * tile_w + x] = val;
    }
    __syncthreads();

    // only threads corresponding to valid output locations perform pooling
    if (h_out < out_H && w_out < out_W && n < N && c < C) {
        int local_h0 = th * stride;
        int local_w0 = tw * stride;
        float max_val = -FLT_MAX;
        // pool over kernel window in shared memory
        for (int dh = 0; dh < kernel_size; ++dh) {
            for (int dw = 0; dw < kernel_size; ++dw) {
                float v = patch[(local_h0 + dh) * tile_w + (local_w0 + dw)];
                if (v > max_val) max_val = v;
            }
        }
        int out_idx = ((n * C + c) * out_H + h_out) * out_W + w_out;
        output[out_idx] = max_val;
    }
}

// External C wrapper calling the optimized kernel
extern "C" void max_pooling_2d_optimized(
    const float* input,
    float* output,
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

    // configure block and grid
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (out_W + blockDim.x - 1) / blockDim.x,
        (out_H + blockDim.y - 1) / blockDim.y,
        N * C
    );
    // shared memory size per block: tile_h * tile_w floats
    int tile_h = blockDim.y * stride + (kernel_size - 1);
    int tile_w = blockDim.x * stride + (kernel_size - 1);
    size_t shmem_bytes = sizeof(float) * tile_h * tile_w;

    // launch optimized kernel
    max_pooling_2d_kernel_optimized<<<gridDim, blockDim, shmem_bytes>>>(
        input, output, N, C, H, W, out_H, out_W, kernel_size, stride, padding
    );
    cudaDeviceSynchronize();
}