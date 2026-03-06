#include <cuda.h>
#include <cuda_runtime.h>

// Maximum supported columns for constant memory
constexpr int MAX_COLS = 2048;

// Copy of dense vector x in constant memory for low-latency access
__constant__ float const_x[MAX_COLS];

// Optimized COO SpMV kernel using read-only cache and constant cache
__global__ void spmv_coo_kernel_optimized(
    int nnz,
    const float* __restrict__ values,
    const int*   __restrict__ row_indices,
    const int*   __restrict__ col_indices,
    float* y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        // Load through read-only cache
        float v = __ldg(values + idx);
        int   r = __ldg(row_indices + idx);
        int   c = __ldg(col_indices + idx);
        // Fetch x from constant cache and accumulate
        atomicAdd(&y[r], v * const_x[c]);
    }
}

// External C wrapper with the same signature as original, suffixed _optimized
extern "C" void spmv_coo_optimized(
    int    rows,
    int    nnz,
    const float* values,
    const int*   row_indices,
    const int*   col_indices,
    const float* x,
    float*       y)
{
    // Copy the full dense vector (length == MAX_COLS == 2048) into constant memory
    // x is a device pointer, so use cudaMemcpyDeviceToDevice
    constexpr size_t const_capacity = MAX_COLS * sizeof(float);
    cudaMemcpyToSymbol(const_x, x, const_capacity, /*offset=*/0,
                       cudaMemcpyDeviceToDevice);

    // Launch configuration: 256 threads per block
    dim3 block(256, 1, 1);
    dim3 grid((nnz + block.x - 1) / block.x, 1, 1);

    spmv_coo_kernel_optimized<<<grid, block>>>(
        nnz, values, row_indices, col_indices, y);
    cudaDeviceSynchronize();
}