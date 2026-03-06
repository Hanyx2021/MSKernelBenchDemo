#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Warp-centric CSR SpMV kernel: one warp per row, coalesced loads, read-only cache
__global__ void csr_spmv_kernel_optimized(
    int rows,
    const int* __restrict__ row_ptr,
    const int* __restrict__ cols,
    const __nv_bfloat16* __restrict__ values,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ y)
{
    const int warp_size = 32;
    int warps_per_block = blockDim.x / warp_size;
    int warp_id_in_block = threadIdx.x / warp_size;
    int warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    if (warp_id >= rows) return;

    int row = warp_id;
    int lane = threadIdx.x & (warp_size - 1);
    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    float sum = 0.0f;
    // Coalesced loads using read-only cache
    for (int idx = row_start + lane; idx < row_end; idx += warp_size) {
        __nv_bfloat16 v_bf = __ldg(&values[idx]);
        int col = __ldg(&cols[idx]);
        __nv_bfloat16 x_bf = __ldg(&x[col]);
        sum += __bfloat162float(v_bf) * __bfloat162float(x_bf);
    }

    // Warp-wide reduction using shuffle
    unsigned mask = 0xffffffff;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Write result by lane 0 of each warp
    if (lane == 0) {
        y[row] = sum;
    }
}

// Convert float array to bfloat16
__global__ void convert_float_to_bf16_kernel_optimized(
    const float* __restrict__ float_array,
    __nv_bfloat16* __restrict__ bf16_array,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmv_csc_bf16_optimized(
    int rows,
    int columns,
    const __nv_bfloat16* d_values,
    const int* d_row_indices,
    const int* d_col_offsets,
    const __nv_bfloat16* d_x,
    __nv_bfloat16* d_y)
{
    // Copy CSC structure from device to host
    int h_nnz;
    int* h_col_offsets = (int*)malloc((columns + 1) * sizeof(int));
    cudaMemcpy(h_col_offsets, d_col_offsets, (columns + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    h_nnz = h_col_offsets[columns];

    int* h_row_indices = (int*)malloc(h_nnz * sizeof(int));
    __nv_bfloat16* h_values = (__nv_bfloat16*)malloc(h_nnz * sizeof(__nv_bfloat16));
    cudaMemcpy(h_row_indices, d_row_indices, h_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, d_values, h_nnz * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Build CSR on host
    int* h_row_counts = (int*)calloc(rows, sizeof(int));
    for (int i = 0; i < h_nnz; ++i) {
        int r = h_row_indices[i];
        h_row_counts[r]++;
    }
    int* h_csr_row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    h_csr_row_ptr[0] = 0;
    for (int i = 0; i < rows; ++i) {
        h_csr_row_ptr[i + 1] = h_csr_row_ptr[i] + h_row_counts[i];
    }
    int* h_csr_cols    = (int*)malloc(h_nnz * sizeof(int));
    __nv_bfloat16* h_csr_values = (__nv_bfloat16*)malloc(h_nnz * sizeof(__nv_bfloat16));
    int* h_position    = (int*)malloc(rows * sizeof(int));
    memcpy(h_position, h_csr_row_ptr, rows * sizeof(int));

    for (int col = 0; col < columns; ++col) {
        int start = h_col_offsets[col];
        int end   = h_col_offsets[col + 1];
        for (int j = start; j < end; ++j) {
            int row = h_row_indices[j];
            int dest = h_position[row]++;
            h_csr_cols[dest]    = col;
            h_csr_values[dest]  = h_values[j];
        }
    }

    // Allocate and copy CSR to device
    int* d_csr_row_ptr = nullptr;
    int* d_csr_cols    = nullptr;
    __nv_bfloat16* d_csr_values = nullptr;
    cudaMalloc(&d_csr_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_csr_cols,    h_nnz * sizeof(int));
    cudaMalloc(&d_csr_values,  h_nnz * sizeof(__nv_bfloat16));
    cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_cols,    h_csr_cols,    h_nnz * sizeof(int),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_values,  h_csr_values,  h_nnz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Free host CSR temp arrays
    free(h_col_offsets);
    free(h_row_indices);
    free(h_values);
    free(h_row_counts);
    free(h_csr_row_ptr);
    free(h_csr_cols);
    free(h_csr_values);
    free(h_position);

    // Allocate output float vector
    float* d_y_float = nullptr;
    cudaMalloc(&d_y_float, rows * sizeof(float));
    cudaMemset(d_y_float, 0, rows * sizeof(float));

    // Launch warp-centric CSR SpMV kernel
    const int warps_per_block = 4;
    int block_size = warps_per_block * 32;
    int grid_size  = (rows + warps_per_block - 1) / warps_per_block;
    csr_spmv_kernel_optimized<<<grid_size, block_size>>>(
        rows,
        d_csr_row_ptr,
        d_csr_cols,
        d_csr_values,
        d_x,
        d_y_float
    );

    // Convert float result to bfloat16
    int conv_grid = (rows + block_size - 1) / block_size;
    convert_float_to_bf16_kernel_optimized<<<conv_grid, block_size>>>(
        d_y_float,
        d_y,
        rows
    );

    // Free device memory
    cudaFree(d_csr_row_ptr);
    cudaFree(d_csr_cols);
    cudaFree(d_csr_values);
    cudaFree(d_y_float);
    cudaDeviceSynchronize();
}
