#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

// Optimized sparse-dense matrix multiplication (CSR format) using read-only cache for X and values
__global__ void spmm_csr_kernel_optimized(
    int rows,
    int K,
    const float* __restrict__ values,
    const int*   __restrict__ col_indices,
    const int*   __restrict__ row_offsets,
    const float* __restrict__ X,
    float* Y) {
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col_k = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col_k < K) {
        float sum = 0.0f;
        int row_start = row_offsets[row];
        int row_end   = row_offsets[row + 1];

        for (int j = row_start; j < row_end; ++j) {
            // Load value and X using read-only cache (__ldg)
            float v = __ldg(values + j);
            int   c = col_indices[j];
            float x = __ldg(X + c * K + col_k);
            sum += v * x;
        }
        Y[row * K + col_k] = sum;
    }
}

extern "C" void spmm_csr_optimized(
    int rows,
    int K,
    const float* values,
    const int*   col_indices,
    const int*   row_offsets,
    const float* X,
    float*       Y) {
    // Launch configuration
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    spmm_csr_kernel_optimized<<<grid, block>>>(
        rows, K,
        values, col_indices, row_offsets,
        X, Y);
    cudaDeviceSynchronize();
}