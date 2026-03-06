#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

// Tile size for K-dimension blocking
#define TILE 256

// Kernel to convert bfloat16 array to float array
__global__ void convert_bf16_to_float_kernel_optimized(
    const __nv_bfloat16* bf16_array,
    float* float_array,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float_array[idx] = __bfloat162float(bf16_array[idx]);
    }
}

// Kernel to convert float array back to bfloat16
__global__ void convert_float_to_bf16_kernel_optimized(
    const float* float_array,
    __nv_bfloat16* bf16_array,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

// CSR-based SpMM kernel operating on float data, one block per row, optimized with shared memory
__global__ void spmm_csr_kernel_optimized(
    int rows,
    int K,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_csr,
    const float* __restrict__ val_f,
    const float* __restrict__ X_f,
    float* Y_f) {
    extern __shared__ char smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int start = row_ptr[row];
    int end   = row_ptr[row + 1];
    int nnz   = end - start;

    // Shared-memory buffers for values and column indices
    float* s_val = (float*)smem;
    int*   s_col = (int*)(smem + nnz * sizeof(float));

    // Load sparse row into shared memory
    int t = threadIdx.x;
    if (t < nnz) {
        s_val[t] = val_f[start + t];
        s_col[t] = col_csr[start + t];
    }
    __syncthreads();

    // Iterate over tiles along K dimension
    for (int tile_start = 0; tile_start < K; tile_start += TILE) {
        int k_index = tile_start + t;
        if (k_index < K) {
            float acc = 0.0f;
            // Accumulate contributions for this tile using shared memory
            for (int j = 0; j < nnz; ++j) {
                float v = s_val[j];
                int   c = s_col[j];
                // Use read-only cache for dense input
                float x = __ldg(&X_f[c * (size_t)K + k_index]);
                acc += v * x;
            }
            Y_f[row * (size_t)K + k_index] = acc;
        }
        // no per-tile __syncthreads() needed
    }
}

extern "C" void spmm_coo_bf16_optimized(
    int rows,
    int nnz,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y) {
    // Copy COO data to host
    std::vector<int> h_row(nnz), h_col(nnz);
    std::vector<__nv_bfloat16> h_val_bf16(nnz);
    cudaMemcpy(h_row.data(), row_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col.data(), col_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val_bf16.data(), values,   nnz * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Build CSR row_ptr with exclusive prefix sum
    std::vector<int> h_row_ptr(rows + 1, 0);
    for (int i = 0; i < nnz; ++i) {
        int r = h_row[i];
        h_row_ptr[r + 1]++;
    }
    for (int i = 1; i <= rows; ++i) {
        h_row_ptr[i] += h_row_ptr[i - 1];
    }

    // Compute maximum non-zeros per row for shared memory sizing
    int max_nnz = 0;
    for (int r = 0; r < rows; ++r) {
        int count = h_row_ptr[r + 1] - h_row_ptr[r];
        if (count > max_nnz) max_nnz = count;
    }
    size_t shared_mem_size = max_nnz * (sizeof(float) + sizeof(int));

    // Scatter into CSR storage and convert values to float
    std::vector<int>   h_col_csr(nnz);
    std::vector<float> h_val_f(nnz);
    std::vector<int>   cur_pos = h_row_ptr;
    for (int i = 0; i < nnz; ++i) {
        int r = h_row[i];
        int pos = cur_pos[r]++;
        h_col_csr[pos] = h_col[i];
        h_val_f[pos]    = __bfloat162float(h_val_bf16[i]);
    }

    // Determine number of columns
    int max_col = 0;
    for (int c : h_col) max_col = std::max(max_col, c);
    int cols = max_col + 1;

    // Allocate and upload CSR arrays to device
    int* d_row_ptr = nullptr;
    int* d_col_csr = nullptr;
    float* d_val_f = nullptr;
    cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_col_csr, nnz * sizeof(int));
    cudaMalloc(&d_val_f,   nnz * sizeof(float));
    cudaMemcpy(d_row_ptr,  h_row_ptr.data(),     (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_csr,  h_col_csr.data(),     nnz * sizeof(int),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_f,    h_val_f.data(),       nnz * sizeof(float),      cudaMemcpyHostToDevice);

    // Convert input dense matrix X to float
    float* d_X_f = nullptr;
    cudaMalloc(&d_X_f, (size_t)cols * K * sizeof(float));
    const int block_convert = 256;
    int total_X = cols * K;
    int grid_convert_X = (total_X + block_convert - 1) / block_convert;
    convert_bf16_to_float_kernel_optimized<<<grid_convert_X, block_convert>>>(
        X, d_X_f, total_X);

    // Allocate output in float
    float* d_Y_f = nullptr;
    cudaMalloc(&d_Y_f, (size_t)rows * K * sizeof(float));

    // Launch CSR-based SpMM kernel with shared memory
    dim3 block(TILE);
    dim3 grid(rows);
    spmm_csr_kernel_optimized<<<grid, block, shared_mem_size>>>(
        rows, K,
        d_row_ptr,
        d_col_csr,
        d_val_f,
        d_X_f,
        d_Y_f);

    // Convert result back to bfloat16
    int total_Y = rows * K;
    int grid_convert_Y = (total_Y + block_convert - 1) / block_convert;
    convert_float_to_bf16_kernel_optimized<<<grid_convert_Y, block_convert>>>(
        d_Y_f, Y, total_Y);

    // Free intermediate buffers
    cudaFree(d_row_ptr);
    cudaFree(d_col_csr);
    cudaFree(d_val_f);
    cudaFree(d_X_f);
    cudaFree(d_Y_f);

    cudaDeviceSynchronize();
}
