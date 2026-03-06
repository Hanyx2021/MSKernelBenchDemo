#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Optimized SPMM ELL-format kernel for bf16 with tuned block shape for higher occupancy
__global__ void spmm_ell_bf16_kernel_optimized(
    int rows,
    int max_nnz_per_row,
    int K,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *X,
    __nv_bfloat16 *Y)
{
    extern __shared__ char smem[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int k   = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory layout: first col_ids, then values
    int *s_col_ids = (int*)smem;
    __nv_bfloat16 *s_vals = (__nv_bfloat16*)(smem + sizeof(int) * blockDim.x * max_nnz_per_row);

    // Cooperative loading: one thread (y==0) per row loads the entire row's indices & values
    if (threadIdx.y == 0 && row < rows) {
        int base_idx        = row * max_nnz_per_row;
        int smem_row_offset = threadIdx.x * max_nnz_per_row;
        for (int e = 0; e < max_nnz_per_row; ++e) {
            int idx = base_idx + e;
            s_col_ids[smem_row_offset + e] = col_ids[idx];
            s_vals[smem_row_offset + e]    = values[idx];
        }
    }
    __syncthreads();

    // Compute the spmm for this (row, k) thread
    if (row < rows && k < K) {
        float sum = 0.0f;
        int smem_row_offset = threadIdx.x * max_nnz_per_row;
        for (int e = 0; e < max_nnz_per_row; ++e) {
            int c = s_col_ids[smem_row_offset + e];
            if (c >= 0) {
                float a_val = __bfloat162float(s_vals[smem_row_offset + e]);
                float x_val = __bfloat162float(X[c * K + k]);
                sum += a_val * x_val;
            }
        }
        Y[row * K + k] = __float2bfloat16(sum);
    }
}

extern "C" void spmm_ell_bf16_optimized(
    int rows,
    int max_nnz_per_row,
    int K,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *X,
    __nv_bfloat16 *Y)
{
    // Tuned block shape: 16 rows per block, 16 K-lanes per block
    dim3 block(16, 16, 1);
    dim3 grid((rows + block.x - 1) / block.x,
              (K    + block.y - 1) / block.y,
              1);
    // Shared memory: ints + bf16 values for 16 rows x max_nnz_per_row
    size_t shared_bytes = (sizeof(int) + sizeof(__nv_bfloat16)) * block.x * max_nnz_per_row;

    spmm_ell_bf16_kernel_optimized<<<grid, block, shared_bytes>>>(
        rows, max_nnz_per_row, K,
        values, col_ids, X, Y);
    cudaDeviceSynchronize();
}