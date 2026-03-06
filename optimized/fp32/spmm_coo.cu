#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>

// Tunable tile sizes for optimized occupancy and coalesced accesses
static constexpr int TILE_R = 16;
static constexpr int TILE_K = 32;

// Row-wise CSR-based SpMM kernel (atomic-free), optimized thread-block shape
__global__ void spmm_coo_kernel_optimized(
    int rows,
    int K,
    const float* __restrict__ values_csr,
    const int* __restrict__ rowPtr,
    const int* __restrict__ col_indices_csr,
    const float* __restrict__ X,
    float* __restrict__ Y)
{
    // threadIdx.x iterates over K for coalesced loads/stores
    int idx_k = blockIdx.x * TILE_K + threadIdx.x;
    // threadIdx.y iterates over rows
    int idx_r = blockIdx.y * TILE_R + threadIdx.y;
    if (idx_r < rows && idx_k < K) {
        float sum = 0.0f;
        int start = rowPtr[idx_r];
        int end   = rowPtr[idx_r + 1];
        // Dot-product over nonzeros in row idx_r
        for (int p = start; p < end; ++p) {
            int c = col_indices_csr[p];
            sum += values_csr[p] * X[c * K + idx_k];
        }
        Y[idx_r * K + idx_k] = sum;
    }
}

extern "C" void spmm_coo_optimized(
    int rows,
    int nnz,
    int K,
    const float* d_values_coo,
    const int* d_row_indices_coo,
    const int* d_col_indices_coo,
    const float* d_X,
    float* d_Y)
{
    // 1) Copy COO data to host for CSR conversion
    std::vector<int>   h_rowIdx(nnz);
    std::vector<int>   h_colIdx(nnz);
    std::vector<float> h_values_coo(nnz);
    cudaMemcpy(h_rowIdx.data(),     d_row_indices_coo, nnz * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colIdx.data(),     d_col_indices_coo, nnz * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values_coo.data(), d_values_coo,      nnz * sizeof(float), cudaMemcpyDeviceToHost);

    // 2) Build CSR rowPtr
    std::vector<int> h_rowPtr(rows + 1, 0);
    for (int i = 0; i < nnz; ++i) {
        int r = h_rowIdx[i];
        h_rowPtr[r + 1]++;
    }
    // Prefix sum
    for (int r = 1; r <= rows; ++r) {
        h_rowPtr[r] += h_rowPtr[r - 1];
    }

    // 3) Allocate host CSR arrays and fill
    std::vector<float> h_values_csr(nnz);
    std::vector<int>   h_colIdx_csr(nnz);
    std::vector<int>   h_offset = h_rowPtr;  // working copy
    for (int i = 0; i < nnz; ++i) {
        int r = h_rowIdx[i];
        int dest = h_offset[r]++;
        h_values_csr[dest] = h_values_coo[i];
        h_colIdx_csr[dest]  = h_colIdx[i];
    }

    // 4) Allocate device CSR arrays
    int*   d_rowPtr;
    float* d_values_csr;
    int*   d_colIdx_csr;
    cudaMalloc(&d_rowPtr,    (rows + 1) * sizeof(int));
    cudaMalloc(&d_values_csr, nnz * sizeof(float));
    cudaMalloc(&d_colIdx_csr, nnz * sizeof(int));

    // 5) Copy CSR arrays to device
    cudaMemcpy(d_rowPtr,     h_rowPtr.data(),    (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_csr, h_values_csr.data(), nnz * sizeof(float),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx_csr, h_colIdx_csr.data(), nnz * sizeof(int),        cudaMemcpyHostToDevice);

    // 6) Initialize output Y to zero
    cudaMemset(d_Y, 0, rows * (size_t)K * sizeof(float));

    // 7) Launch optimized CSR-based SpMM kernel with tuned block shape
    dim3 block(TILE_K, TILE_R, 1);
    dim3 grid((K    + TILE_K - 1) / TILE_K,
              (rows + TILE_R - 1) / TILE_R,
              1);
    spmm_coo_kernel_optimized<<<grid, block>>>(
        rows, K,
        d_values_csr,
        d_rowPtr,
        d_colIdx_csr,
        d_X,
        d_Y);

    cudaDeviceSynchronize();

    // 8) Cleanup temporary device arrays
    cudaFree(d_rowPtr);
    cudaFree(d_values_csr);
    cudaFree(d_colIdx_csr);
}
