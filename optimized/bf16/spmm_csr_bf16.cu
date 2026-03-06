#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

// Optimized CSR-BF16 SpMM Kernel with full-warp coalesced loads
__global__ void spmm_csr_bf16_kernel_optimized(
    int rows,
    int K,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y)
{
    // Each block covers blockDim.y rows and blockDim.x columns (32-wide for coalescing)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_k = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col_k < K) {
        float sum = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];

        for (int j = row_start; j < row_end; j++) {
            int col_index = col_indices[j];
            float val_f = __bfloat162float(values[j]);
            float x_val_f = __bfloat162float(X[col_index * K + col_k]);
            sum += val_f * x_val_f;
        }

        Y[row * K + col_k] = __float2bfloat16(sum);
    }
}

extern "C" void spmm_csr_bf16_optimized(
    int rows,
    int K,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y)
{
    // Launch configuration tuned for RTX 4090: 32 threads in x for full-warps, 8 threads in y
    dim3 block(32, 8);
    dim3 grid((K + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    spmm_csr_bf16_kernel_optimized<<<grid, block>>>(
        rows, K, values, col_indices, row_offsets, X, Y);
    cudaDeviceSynchronize();
}
