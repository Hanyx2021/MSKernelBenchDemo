#include <cuda.h>
#include <cuda_runtime.h>

#define ROWS_PER_BLOCK 16

// ==== OPTIMIZED KERNEL START ====
// Rename the __global__ to avoid name collision and mark as optimized
__global__ void spmv_ell_kernel_optimized(
    int rows,
    int max_nnz_per_row,
    const float *__restrict__ values,
    const int *__restrict__ col_ids,
    const float *__restrict__ x,
    float *y) {
    int lane = threadIdx.x;            // 0..31 within warp
    int row  = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;

    float sum = 0.0f;
    int base = row * max_nnz_per_row;

    // each lane in the warp processes a strided subset of non-zeros
    for (int e = lane; e < max_nnz_per_row; e += 32) {
        int idx = base + e;
        int col = __ldg(col_ids + idx);
        if (col != -1) {
            sum += __ldg(values + idx) * __ldg(x + col);
        }
    }

    // warp-wide reduction by shuffle
    unsigned int full_mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(full_mask, sum, offset);
    }

    // only lane 0 writes the final sum
    if (lane == 0) {
        y[row] = sum;
    }
}

// Host wrapper remains spmv_ell_optimized
extern "C" void spmv_ell_optimized(
    int rows,
    int max_nnz_per_row,
    const float *values,
    const int *col_ids,
    const float *x,
    float *y) {
    // one warp per row, ROWS_PER_BLOCK warps per block
    dim3 block(32, ROWS_PER_BLOCK, 1);
    dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);

    // launch the renamed optimized kernel
    spmv_ell_kernel_optimized<<<grid, block>>>(
        rows, max_nnz_per_row,
        values, col_ids,
        x, y);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====
