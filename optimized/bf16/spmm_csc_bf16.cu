#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <algorithm>

// Configuration struct for optimized kernel launches
typedef struct {
    dim3 spmm_block;
    dim3 spmm_grid;
    dim3 convert_block;
    dim3 convert_grid;
} SpmmCscBf16OptimizedConfig;

// CSR-based SpMM kernel: each thread computes one (row, k) output without atomics
// Warp-aligned mapping: threadIdx.x -> K dimension, threadIdx.y -> row dimension
__global__ void spmm_csr_float_kernel_optimized(
    int rows,
    int K,
    const __nv_bfloat16* __restrict__ values_csr,
    const int* __restrict__ col_indices_csr,
    const int* __restrict__ row_offsets_csr,
    const __nv_bfloat16* __restrict__ X,
    float* __restrict__ Y_float)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && k < K) {
        float sum = 0.0f;
        int start = __ldg(&row_offsets_csr[r]);
        int end   = __ldg(&row_offsets_csr[r + 1]);
        for (int idx = start; idx < end; ++idx) {
            int c = __ldg(&col_indices_csr[idx]);
            float a_val = __bfloat162float(__ldg(&values_csr[idx]));
            float x_val = __bfloat162float(__ldg(&X[c * K + k]));
            sum += a_val * x_val;
        }
        Y_float[r * K + k] = sum;
    }
}

// Optimized conversion from float array to bfloat16 array
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

// External C wrapper for the optimized operator
extern "C" void spmm_csc_bf16_optimized(
    int rows,
    int columns,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y)
{
    // Determine number of non-zeros (nnz)
    int h_nnz = 0;
    cudaMemcpy(&h_nnz, col_offsets + columns, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy CSC data to host
    std::vector<int> h_col_offsets(columns + 1);
    cudaMemcpy(h_col_offsets.data(), col_offsets, sizeof(int) * (columns + 1), cudaMemcpyDeviceToHost);
    std::vector<int> h_row_indices(h_nnz);
    cudaMemcpy(h_row_indices.data(), row_indices, sizeof(int) * h_nnz, cudaMemcpyDeviceToHost);
    std::vector<__nv_bfloat16> h_values(h_nnz);
    cudaMemcpy(h_values.data(), values, sizeof(__nv_bfloat16) * h_nnz, cudaMemcpyDeviceToHost);

    // Build CSR structure on host
    std::vector<int> h_row_offsets(rows + 1, 0);
    for (int col = 0; col < columns; ++col) {
        int start = h_col_offsets[col];
        int end   = h_col_offsets[col + 1];
        for (int j = start; j < end; ++j) {
            int r = h_row_indices[j];
            h_row_offsets[r + 1]++;
        }
    }
    for (int r = 1; r <= rows; ++r) {
        h_row_offsets[r] += h_row_offsets[r - 1];
    }
    
    std::vector<int> h_row_ptr = h_row_offsets;
    std::vector<int> h_col_indices_csr(h_nnz);
    std::vector<__nv_bfloat16> h_values_csr(h_nnz);
    for (int col = 0; col < columns; ++col) {
        int start = h_col_offsets[col];
        int end   = h_col_offsets[col + 1];
        for (int j = start; j < end; ++j) {
            int r = h_row_indices[j];
            int pos = h_row_ptr[r]++;
            h_col_indices_csr[pos] = col;
            h_values_csr[pos]      = h_values[j];
        }
    }

    // Allocate and copy CSR arrays to device
    int* d_row_offsets_csr = nullptr;
    int* d_col_indices_csr = nullptr;
    __nv_bfloat16* d_values_csr = nullptr;
    cudaMalloc(&d_row_offsets_csr, sizeof(int) * (rows + 1));
    cudaMalloc(&d_col_indices_csr, sizeof(int) * h_nnz);
    cudaMalloc(&d_values_csr, sizeof(__nv_bfloat16) * h_nnz);
    cudaMemcpy(d_row_offsets_csr, h_row_offsets.data(), sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices_csr, h_col_indices_csr.data(), sizeof(int) * h_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_csr, h_values_csr.data(), sizeof(__nv_bfloat16) * h_nnz, cudaMemcpyHostToDevice);

    // Allocate and zero temporary float output
    float* d_Y_float = nullptr;
    size_t total_floats = static_cast<size_t>(rows) * static_cast<size_t>(K);
    cudaMalloc(&d_Y_float, total_floats * sizeof(float));
    cudaMemset(d_Y_float, 0, total_floats * sizeof(float));

    // Configure and launch optimized spmm CSR kernel with warp-aligned mapping
    SpmmCscBf16OptimizedConfig cfg;
    cfg.spmm_block    = dim3(32, 8);
    cfg.spmm_grid     = dim3((K    + cfg.spmm_block.x - 1) / cfg.spmm_block.x,
                              (rows + cfg.spmm_block.y - 1) / cfg.spmm_block.y);
    cfg.convert_block = dim3(256);
    cfg.convert_grid  = dim3((total_floats + cfg.convert_block.x - 1) / cfg.convert_block.x);

    spmm_csr_float_kernel_optimized<<<cfg.spmm_grid, cfg.spmm_block>>>(
        rows, K,
        d_values_csr, d_col_indices_csr, d_row_offsets_csr,
        X, d_Y_float);

    // Convert accumulated floats back to bfloat16
    convert_float_to_bf16_kernel_optimized<<<cfg.convert_grid, cfg.convert_block>>>(
        d_Y_float, Y, static_cast<int>(total_floats));

    // Clean up and synchronize
    cudaFree(d_Y_float);
    cudaFree(d_row_offsets_csr);
    cudaFree(d_col_indices_csr);
    cudaFree(d_values_csr);
    cudaDeviceSynchronize();
}