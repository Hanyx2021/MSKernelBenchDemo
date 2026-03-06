#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for rows per block
constexpr int ROWS_PER_BLOCK = 16;
// Unroll factor for inner loop
constexpr int UNROLL_FACTOR = 4;

// Optimized ELLPACK SpMM kernel using shared memory, thread-dimension swap,
// branch-free accumulation, and inner-loop unrolling for higher ILP
__global__ void spmm_ell_kernel_optimized(
    int rows,
    int max_nnz_per_row,
    int K,
    const float *values,
    const int *col_ids,
    const float * __restrict__ X,
    float *Y)
{
    extern __shared__ char smem[];
    // Shared memory layout: first values, then col_ids
    float *s_values = reinterpret_cast<float*>(smem);
    int   *s_colIds = reinterpret_cast<int*>(smem + sizeof(float) * ROWS_PER_BLOCK * max_nnz_per_row);

    // Cooperative load of ELL data into shared memory with padding
    int totalElems = ROWS_PER_BLOCK * max_nnz_per_row;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    while (tid < totalElems) {
        int rowLocal  = tid / max_nnz_per_row;
        int e         = tid % max_nnz_per_row;
        int rowGlobal = blockIdx.x * ROWS_PER_BLOCK + rowLocal;
        int idxLocal  = rowLocal * max_nnz_per_row + e;
        int idxGlobal = rowGlobal * max_nnz_per_row + e;
        int col = (rowGlobal < rows) ? col_ids[idxGlobal] : -1;
        bool valid = (col >= 0);
        s_colIds[idxLocal]  = valid ? col : 0;
        s_values[idxLocal]  = valid ? values[idxGlobal] : 0.0f;
        tid += blockDim.x * blockDim.y;
    }
    __syncthreads();

    // Compute the output for each (rowLocal, kLocal)
    int rowLocal  = threadIdx.y;
    int kLocal    = threadIdx.x;
    int rowGlobal = blockIdx.x * ROWS_PER_BLOCK + rowLocal;
    int kGlobal   = blockIdx.y * blockDim.x + kLocal;

    if (rowGlobal < rows && kGlobal < K) {
        int baseIdx = rowLocal * max_nnz_per_row;
        // Multiple accumulators for higher ILP
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        int e = 0;
        // Unrolled loop in chunks of UNROLL_FACTOR
        for (; e + (UNROLL_FACTOR - 1) < max_nnz_per_row; e += UNROLL_FACTOR) {
#pragma unroll
            {
                int idx0 = baseIdx + e;
                int idx1 = idx0 + 1;
                int idx2 = idx0 + 2;
                int idx3 = idx0 + 3;
                float v0 = s_values[idx0];  int c0 = s_colIds[idx0];
                float v1 = s_values[idx1];  int c1 = s_colIds[idx1];
                float v2 = s_values[idx2];  int c2 = s_colIds[idx2];
                float v3 = s_values[idx3];  int c3 = s_colIds[idx3];
                float x0 = __ldg(&X[c0 * K + kGlobal]);
                float x1 = __ldg(&X[c1 * K + kGlobal]);
                float x2 = __ldg(&X[c2 * K + kGlobal]);
                float x3 = __ldg(&X[c3 * K + kGlobal]);
                acc0 += v0 * x0;
                acc1 += v1 * x1;
                acc2 += v2 * x2;
                acc3 += v3 * x3;
            }
        }
        // Handle remaining elements
        for (; e < max_nnz_per_row; ++e) {
            int idx = baseIdx + e;
            float v = s_values[idx];
            int c   = s_colIds[idx];
            float x = __ldg(&X[c * K + kGlobal]);
            acc0   += v * x;
        }
        // Reduce accumulators and store result
        float sum = acc0 + acc1 + acc2 + acc3;
        Y[rowGlobal * K + kGlobal] = sum;
    }
}

extern "C" void spmm_ell_optimized(
    int rows,
    int max_nnz_per_row,
    int K,
    const float *values,
    const int *col_ids,
    const float *X,
    float *Y)
{
    // Configure block and grid dimensions
    dim3 block(32, ROWS_PER_BLOCK, 1);
    dim3 grid(
        (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK,
        (K    + block.x         - 1) / block.x,
        1);

    // Shared memory size for values and column indices
    size_t sharedMemSize = (sizeof(float) + sizeof(int)) * ROWS_PER_BLOCK * max_nnz_per_row;

    // Launch optimized kernel
    spmm_ell_kernel_optimized<<<grid, block, sharedMemSize>>>(
        rows,
        max_nnz_per_row,
        K,
        values,
        col_ids,
        X,
        Y);
    cudaDeviceSynchronize();
}
