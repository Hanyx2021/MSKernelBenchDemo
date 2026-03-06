#include <cuda.h>
#include <cuda_runtime.h>

// Configuration structure for optimized launch parameters
struct VectorAddConfigOptimized {
    int threads_per_block;
    int blocks;
};

// Optimized vector addition kernel with vectorized memory access (float4)
__global__ void vector_add_kernel_optimized(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int num_vec,
                                            int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized processing: each thread handles one float4 (4 elements)
    if (tid < num_vec) {
        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* B4 = reinterpret_cast<const float4*>(B);
        float4* C4 = reinterpret_cast<float4*>(C);

        float4 a = A4[tid];
        float4 b = B4[tid];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[tid] = c;
    }

    // Tail-cleanup for remaining elements (N % 4)
    int idx = num_vec * 4 + tid;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void vector_add_optimized(const float* A,
                                       const float* B,
                                       float* C,
                                       int N) {
    int num_vec = N / 4;

    // Block-size tuning: use smaller blocks to increase occupancy
    VectorAddConfigOptimized config;
    config.threads_per_block = 128;
    config.blocks = (num_vec + config.threads_per_block - 1) / config.threads_per_block;
    if (config.blocks == 0) config.blocks = 1;

    dim3 block(config.threads_per_block, 1, 1);
    dim3 grid(config.blocks, 1, 1);

    vector_add_kernel_optimized<<<grid, block>>>(A, B, C, num_vec, N);
    cudaDeviceSynchronize();
}