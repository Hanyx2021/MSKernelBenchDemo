#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Optimized kernel with block-level reduction in shared memory
template <unsigned int blockSize>
__global__ void monte_carlo_int_kernel_optimized(
    const float* __restrict__ y_samples,
    float* partial_sum,
    float a,
    float b,
    int N
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Compute per-thread contribution
    float contrib = 0.0f;
    if (idx < (unsigned int)N) {
        contrib = (b - a) * y_samples[idx] / N;
    }
    sdata[tid] = contrib;
    __syncthreads();

    // Intra-block reduction
    if (blockSize >= 1024) {
        if (tid < 512) sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    // Warp-level unrolled reduction (no __syncthreads needed within a warp)
    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // First thread in block adds to global accumulator
    if (tid == 0) {
        atomicAdd(partial_sum, sdata[0]);
    }
}

extern "C" void monte_carlo_int_optimized(
    const float* y_samples,
    float* result,
    float a,
    float b,
    int N
) {
    const int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_bytes = threads_per_block * sizeof(float);

    // Launch optimized kernel with shared memory
    monte_carlo_int_kernel_optimized<threads_per_block><<<blocks_per_grid, threads_per_block, shared_mem_bytes>>>(
        y_samples,
        result,
        a,
        b,
        N
    );

    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in monte_carlo_int_optimized: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
