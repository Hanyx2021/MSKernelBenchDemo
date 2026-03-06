#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>

// Configuration for optimized transpose
struct TransposeBf16OptimizedConfig {
    static constexpr int TILE_DIM = 32;
    static constexpr int BLOCK_ROWS = 8;
};

// Optimized tiled shared-memory transpose kernel for bfloat16
__global__ void matrix_transpose_bf16_kernel_optimized(
    const __nv_bfloat16* A,
    __nv_bfloat16* B,
    int M,
    int N) {
    // Shared tile with padding to avoid bank conflicts
    __shared__ __nv_bfloat16 tile[TransposeBf16OptimizedConfig::TILE_DIM]
                                     [TransposeBf16OptimizedConfig::TILE_DIM + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_dim = TransposeBf16OptimizedConfig::TILE_DIM;
    int block_rows = TransposeBf16OptimizedConfig::BLOCK_ROWS;

    // Load data from global memory to shared tile
    for (int i = 0; i < tile_dim; i += block_rows) {
        int row = by * tile_dim + ty + i;
        int col = bx * tile_dim + tx;
        if (row < M && col < N) {
            tile[ty + i][tx] = A[row * N + col];
        }
    }
    __syncthreads();

    // Write transposed data from shared tile back to global memory
    for (int i = 0; i < tile_dim; i += block_rows) {
        int row = bx * tile_dim + ty + i;  // swapped indices for transpose
        int col = by * tile_dim + tx;
        if (row < N && col < M) {
            B[row * M + col] = tile[tx][ty + i];
        }
    }
    __syncthreads();
}

extern "C" void matrix_transpose_bf16_optimized(
    const __nv_bfloat16* A,
    __nv_bfloat16* B,
    int M,
    int N) {
    // Configure block and grid dimensions
    const int TILE_DIM = TransposeBf16OptimizedConfig::TILE_DIM;
    const int BLOCK_ROWS = TransposeBf16OptimizedConfig::BLOCK_ROWS;

    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (M + TILE_DIM - 1) / TILE_DIM,
              1);

    // Launch optimized kernel
    matrix_transpose_bf16_kernel_optimized<<<grid, block>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
