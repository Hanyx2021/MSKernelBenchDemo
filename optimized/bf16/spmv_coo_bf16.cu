#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// CSR-based SpMV kernel: one thread per row, no atomics
__global__ void spmv_csr_bf16_kernel_optimized(
    int rows,
    int nnz,
    const __nv_bfloat16* values_csr,
    const int* row_offsets,
    const int* col_indices_csr,
    const __nv_bfloat16* x,
    float* y_float) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int start = row_offsets[row];
        int end = row_offsets[row + 1];
        float sum = 0.0f;
        for (int idx = start; idx < end; ++idx) {
            float val = __bfloat162float(values_csr[idx]);
            int col = col_indices_csr[idx];
            float x_val = __bfloat162float(x[col]);
            sum += val * x_val;
        }
        y_float[row] = sum;
    }
}

// Convert float array to bfloat16
__global__ void convert_float_to_bf16_kernel_optimized(
    const float* float_array,
    __nv_bfloat16* bf16_array,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

// Forward-declare the COO->float kernel so the compiler knows about it
__global__ void spmv_coo_float_kernel(
    int nnz,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* x,
    float* y_float);

extern "C" void spmv_coo_bf16_optimized(
    int rows,
    int nnz,
    const __nv_bfloat16* values,    // nnz-long COO values
    const int* row_indices,         // nnz-long COO row indices
    const int* col_indices,         // nnz-long COO col indices
    const __nv_bfloat16* x,
    __nv_bfloat16* y) {
    // Allocate intermediate float output
    float* d_y_float = nullptr;
    cudaMalloc(&d_y_float, rows * sizeof(float));
    cudaMemset(d_y_float, 0, rows * sizeof(float));

    // Launch COO-based SpMV kernel with atomicAdd
    constexpr int BLOCK = 256;
    int grid = (nnz + BLOCK - 1) / BLOCK;
    spmv_coo_float_kernel<<<grid, BLOCK>>>(
        nnz,
        values,
        row_indices,
        col_indices,
        x,
        d_y_float
    );

    // Convert float results back to bfloat16
    constexpr int CV_BLOCK = 256;
    int cv_grid = (rows + CV_BLOCK - 1) / CV_BLOCK;
    convert_float_to_bf16_kernel_optimized<<<cv_grid, CV_BLOCK>>>(
        d_y_float,
        y,
        rows);

    // Clean up and synchronize
    cudaFree(d_y_float);
    cudaDeviceSynchronize();
}