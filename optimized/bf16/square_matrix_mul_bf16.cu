#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TILE 32

// Renamed device kernel to avoid collision with host wrapper symbol
__global__ void square_matrix_mul_bf16_optimized_kernel(
    __nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float accum = 0.0f;

    __shared__ __nv_bfloat16 A_sub[TILE][TILE];
    __shared__ __nv_bfloat16 B_sub[TILE][TILE];

    for (int tile_k = 0; tile_k < M; tile_k += TILE) {
        int a_row = row;
        int a_col = tile_k + threadIdx.x;
        if (a_row < M && a_col < M) {
            A_sub[threadIdx.y][threadIdx.x] = A[a_row * M + a_col];
        } else {
            A_sub[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        int b_row = tile_k + threadIdx.y;
        int b_col = col;
        if (b_row < M && b_col < M) {
            B_sub[threadIdx.y][threadIdx.x] = B[b_row * M + b_col];
        } else {
            B_sub[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a_val = __bfloat162float(A_sub[threadIdx.y][k]);
            float b_val = __bfloat162float(B_sub[k][threadIdx.x]);
            accum += a_val * b_val;
        }

        __syncthreads();
    }

    if (row < M && col < M) {
        C[row * M + col] = __float2bfloat16(accum);
    }
}

// Host wrapper retains the original symbol for the test harness
extern "C" void square_matrix_mul_bf16_optimized(
    __nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M)
{
    dim3 block(TILE, TILE, 1);
    dim3 grid((M + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
    // Launch the renamed device kernel
    square_matrix_mul_bf16_optimized_kernel<<<grid, block>>>(A, B, C, M);
    cudaDeviceSynchronize();
}
