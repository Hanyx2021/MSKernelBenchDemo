#include <cuda.h>
#include <cuda_runtime.h>

// Two-level tiled matrix multiplication configuration with register-blocking
struct MatrixMulOptimizedConfig {
    static constexpr int TILE_M_block = 32;
    static constexpr int TILE_N_block = 32;
    static constexpr int TILE_K_block = 32;
    static constexpr int rM = 2;
    static constexpr int rN = 2;
};

// Kernel optimized with two-level tiling and register-blocked accumulation
__global__ void matrix_mul_kernel_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {
    constexpr int TILE_M = MatrixMulOptimizedConfig::TILE_M_block;
    constexpr int TILE_N = MatrixMulOptimizedConfig::TILE_N_block;
    constexpr int TILE_K = MatrixMulOptimizedConfig::TILE_K_block;
    constexpr int rM = MatrixMulOptimizedConfig::rM;
    constexpr int rN = MatrixMulOptimizedConfig::rN;

    // Shared-memory tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    // Compute starting indices of the block in global matrix
    int blockRowStart = blockIdx.y * TILE_M;
    int blockColStart = blockIdx.x * TILE_N;

    // Local offsets for this thread's register tile
    int localRow = threadIdx.y * rM;
    int localCol = threadIdx.x * rN;

    // Register accumulators for the rM x rN sub-block
    float reg_c[rM][rN];
#pragma unroll
    for (int i = 0; i < rM; ++i) {
#pragma unroll
        for (int j = 0; j < rN; ++j) {
            reg_c[i][j] = 0.0f;
        }
    }

    int numTiles = (K + TILE_K - 1) / TILE_K;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int sA_size = TILE_M * TILE_K;
    int sB_size = TILE_K * TILE_N;

    for (int t = 0; t < numTiles; ++t) {
        int aTileColStart = t * TILE_K;
        // Load A sub-tile into shared memory
        for (int idx = tid; idx < sA_size; idx += blockDim.x * blockDim.y) {
            int rowA = idx / TILE_K;
            int colA = idx % TILE_K;
            int globalRow = blockRowStart + rowA;
            int globalCol = aTileColStart + colA;
            if (globalRow < M && globalCol < K)
                As[rowA][colA] = A[globalRow * K + globalCol];
            else
                As[rowA][colA] = 0.0f;
        }
        // Load B sub-tile into shared memory
        for (int idx = tid; idx < sB_size; idx += blockDim.x * blockDim.y) {
            int rowB = idx / TILE_N;
            int colB = idx % TILE_N;
            int globalRow = aTileColStart + rowB;
            int globalCol = blockColStart + colB;
            if (globalRow < K && globalCol < N)
                Bs[rowB][colB] = B[globalRow * N + globalCol];
            else
                Bs[rowB][colB] = 0.0f;
        }
        __syncthreads();

        // Compute register-blocked accumulations
#pragma unroll
        for (int kIdx = 0; kIdx < TILE_K; ++kIdx) {
            float a0 = As[localRow + 0][kIdx];
            float a1 = As[localRow + 1][kIdx];
            float b0 = Bs[kIdx][localCol + 0];
            float b1 = Bs[kIdx][localCol + 1];
            reg_c[0][0] += a0 * b0;
            reg_c[0][1] += a0 * b1;
            reg_c[1][0] += a1 * b0;
            reg_c[1][1] += a1 * b1;
        }
        __syncthreads();
    }

    // Write back the computed sub-block to global memory
#pragma unroll
    for (int i = 0; i < rM; ++i) {
#pragma unroll
        for (int j = 0; j < rN; ++j) {
            int globalRow = blockRowStart + localRow + i;
            int globalCol = blockColStart + localCol + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = reg_c[i][j];
            }
        }
    }
}

// External C wrapper for the optimized matrix multiplication
extern "C" void matrix_mul_optimized(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K) {
    constexpr int TILE_M = MatrixMulOptimizedConfig::TILE_M_block;
    constexpr int TILE_N = MatrixMulOptimizedConfig::TILE_N_block;
    constexpr int rM = MatrixMulOptimizedConfig::rM;
    constexpr int rN = MatrixMulOptimizedConfig::rN;
    dim3 block(TILE_N / rN, TILE_M / rM, 1);
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M,
        1);
    matrix_mul_kernel_optimized<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}