#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 32

// Optimized tiled shared-memory matrix multiplication kernel
__global__ void square_matrix_mul_kernel_optimized(float* A, float* B, float* C, int M) {
    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (M + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < M)
            A_tile[threadIdx.y][threadIdx.x] = A[row * M + a_col];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE + threadIdx.y;
        if (b_row < M && col < M)
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * M + col];
        else
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}

// External C wrapper maintaining original interface, with optimized kernel launch
extern "C" void square_matrix_mul_optimized(float* A, float* B, float* C, int M) {
    dim3 block(TILE, TILE);
    dim3 grid((M + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    square_matrix_mul_kernel_optimized<<<grid, block>>>(A, B, C, M);
    cudaDeviceSynchronize();
}