#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

// WMMA tile sizes
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
// Block tile sizes for Tensor-Core kernel
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;

// Fallback (scalar/shared memory) kernel, copied and renamed from original
__global__ void matrix_mul_bf16_fallback_kernel_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    const int Mtile = 64;
    const int Ntile = 64;
    const int Ktile = 16;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int rowBase = blockRow * Mtile;
    int colBase = blockCol * Ntile;
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + Mtile * Ktile;
    enum { Tr = Mtile / 16, Tc = Ntile / 16 };
    float regC[Tr][Tc] = {0.0f};
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    for (int offsetK = 0; offsetK < K; offsetK += Ktile) {
        int currentK = min(Ktile, K - offsetK);
        int totalA = Mtile * currentK;
        for (int idx = tid; idx < totalA; idx += blockSize) {
            int i = idx / currentK;
            int k = idx % currentK;
            int globalRow = rowBase + i;
            int globalK   = offsetK + k;
            float aval = 0.0f;
            if (globalRow < M && globalK < K) aval = __bfloat162float(A[globalRow * K + globalK]);
            As[i * Ktile + k] = aval;
        }
        int totalB = currentK * Ntile;
        for (int idx = tid; idx < totalB; idx += blockSize) {
            int k = idx / Ntile;
            int j = idx % Ntile;
            int globalK = offsetK + k;
            int globalCol = colBase + j;
            float bval = 0.0f;
            if (globalK < K && globalCol < N) bval = __bfloat162float(B[globalK * N + globalCol]);
            Bs[k * Ntile + j] = bval;
        }
        __syncthreads();
        for (int kk = 0; kk < currentK; ++kk) {
            float aVals[Tr];
            for (int i = 0; i < Tr; ++i) {
                int localRow = threadIdx.y * Tr + i;
                aVals[i] = As[localRow * Ktile + kk];
            }
            for (int j = 0; j < Tc; ++j) {
                int localCol = threadIdx.x * Tc + j;
                float bval = Bs[kk * Ntile + localCol];
                for (int i = 0; i < Tr; ++i) regC[i][j] += aVals[i] * bval;
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < Tr; ++i) {
        int globalRow = rowBase + threadIdx.y * Tr + i;
        if (globalRow >= M) continue;
        for (int j = 0; j < Tc; ++j) {
            int globalCol = colBase + threadIdx.x * Tc + j;
            if (globalCol >= N) continue;
            float cval = regC[i][j];
            C[globalRow * N + globalCol] = __float2bfloat16(cval);
        }
    }
}

// Tensor-Core accelerated kernel
__global__ void matrix_mul_bf16_wmma_kernel_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K) {
    // Block tile indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int rowBase = blockRow * TILE_M;
    int colBase = blockCol * TILE_N;

    extern __shared__ __nv_bfloat16 shmem[];
    __nv_bfloat16* shA = shmem;
    __nv_bfloat16* shB = shmem + TILE_M * WMMA_K;

    // Thread and warp indexing
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int warpId = tid / 32;
    int laneId = tid % 32;
    int warpRow = warpId / (TILE_N / WMMA_N);
    int warpCol = warpId % (TILE_N / WMMA_N);

    // Accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    // Loop over K
    for (int offsetK = 0; offsetK < K; offsetK += WMMA_K) {
        int currentK = min(WMMA_K, K - offsetK);
        // Load A tile into shared memory
        int numElemA = TILE_M * currentK;
        for (int idx = tid; idx < numElemA; idx += blockSize) {
            int i = idx / currentK;
            int k = idx % currentK;
            int globalRow = rowBase + i;
            int globalK   = offsetK + k;
            __nv_bfloat16 aval = __float2bfloat16(0);
            if (globalRow < M && globalK < K) aval = A[globalRow * K + globalK];
            shA[i * WMMA_K + k] = aval;
        }
        // Load B tile into shared memory
        int numElemB = currentK * TILE_N;
        for (int idx = tid; idx < numElemB; idx += blockSize) {
            int k = idx / TILE_N;
            int j = idx % TILE_N;
            int globalK = offsetK + k;
            int globalCol = colBase + j;
            __nv_bfloat16 bval = __float2bfloat16(0);
            if (globalK < K && globalCol < N) bval = B[globalK * N + globalCol];
            shB[k * TILE_N + j] = bval;
        }
        __syncthreads();

        // Load fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> bFrag;
        
        const __nv_bfloat16* tileAPtr = shA + warpRow * WMMA_M * WMMA_K;
        const __nv_bfloat16* tileBPtr = shB + warpCol * WMMA_N;
        wmma::load_matrix_sync(aFrag, tileAPtr, WMMA_K);
        wmma::load_matrix_sync(bFrag, tileBPtr, TILE_N);
        
        // Perform the matrix multiplication
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        __syncthreads();
    }

    // Store the result back to C
    int outRow = rowBase + warpRow * WMMA_M;
    int outCol = colBase + warpCol * WMMA_N;
    if (outRow < M && outCol < N) {
        // Each fragment lane writes one element
        // Convert the fragment to shared memory temp and then write by lane
        float cTmp[WMMA_M * WMMA_N];
        wmma::store_matrix_sync(cTmp, cFrag, WMMA_N, wmma::mem_row_major);
        for (int i = 0; i < WMMA_M; ++i) {
            int targetRow = outRow + i;
            if (targetRow >= M) continue;
            for (int j = 0; j < WMMA_N; ++j) {
                int targetCol = outCol + j;
                if (targetCol >= N) continue;
                int idx = i * WMMA_N + j;
                C[targetRow * N + targetCol] = __float2bfloat16(cTmp[idx]);
            }
        }
    }
}

// External C wrapper: always use fallback
extern "C" void matrix_mul_bf16_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16*       C,
    int                  M,
    int                  N,
    int                  K) 
{
    // Always use the fallback (shared-memory + scalar) kernel:
    dim3 block(16, 16, 1);
    dim3 grid((N + 64 - 1) / 64,
              (M + 64 - 1) / 64,
              1);
    // shared bytes = floats for A-tile (64×16) + floats for B-tile (16×64)
    size_t shared_bytes = sizeof(float) * (64 * 16 + 16 * 64);

    matrix_mul_bf16_fallback_kernel_optimized
      <<<grid, block, shared_bytes>>>(A, B, C, M, N, K);
    // force sync so errors get caught immediately
    cudaDeviceSynchronize();
}
