#include <cuda.h>
#include <cuda_runtime.h>

// Optimized Simpson integration kernel with block-wide reduction
__global__ void simpson_int_kernel_optimized(
    const float* y_samples,
    float* partial_sum,
    float a,
    float b,
    int N
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int numSegments = (N - 1) / 2;
    float h = (b - a) / (N - 1);
    float thread_sum = 0.0f;

    // Each thread computes one Simpson segment if in range
    if (gid < numSegments) {
        int i = 2 * gid;
        float f0 = y_samples[i];
        float f1 = y_samples[i + 1];
        float f2 = y_samples[i + 2];
        thread_sum = (h / 3.0f) * (f0 + 4.0f * f1 + f2);
    }

    // Store local sum into shared memory
    sdata[tid] = thread_sum;
    __syncthreads();

    // Block-wide reduction in shared memory
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Single atomic add per block
    if (tid == 0) {
        atomicAdd(partial_sum, sdata[0]);
    }
}

extern "C" void simpson_int_optimized(
    const float* y_samples,
    float* result,
    float a,
    float b,
    int N
) {
    const int threads_per_block = 256;
    int numSegments = (N - 1) / 2;
    int blocks_per_grid = (numSegments + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Initialize output accumulator to zero
    cudaMemset(result, 0, sizeof(float));

    // Launch optimized kernel
    simpson_int_kernel_optimized<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        y_samples, result, a, b, N
    );
    cudaDeviceSynchronize();
}