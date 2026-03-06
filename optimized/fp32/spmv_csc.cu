#include <cuda.h>
#include <cuda_runtime.h>

// Optimized Warp-cooperative CSC SpMV kernel using read-only cache
__global__ void spmv_csc_kernel_optimized(
    int columns,
    const float* __restrict__ values,
    const int* __restrict__ row_indices,
    const int* __restrict__ col_offsets,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    // Global thread and lane identifiers
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;            // threadIdx.x % warpSize
    int warp_id = tid >> 5;                    // tid / warpSize

    int col = warp_id;
    if (col >= columns) return;

    // Load x[col] through read-only cache and broadcast across warp
    float x_val = __ldg(&x[col]);
    x_val = __shfl_sync(0xFFFFFFFF, x_val, 0);

    // Load column offsets through read-only cache
    int start = __ldg(&col_offsets[col]);
    int end   = __ldg(&col_offsets[col + 1]);

    // Process non-zeros in chunks of warpSize for coalesced access
    for (int k = start; k < end; k += 32) {
        int idx = k + lane_id;
        if (idx < end) {
            // Load sparse data through read-only cache
            float v = __ldg(&values[idx]);
            int   r = __ldg(&row_indices[idx]);
            atomicAdd(&y[r], v * x_val);
        }
    }
}

extern "C" void spmv_csc_optimized(
    int rows,
    int columns,
    const float* values,
    const int* row_indices,
    const int* col_offsets,
    const float* x,
    float* y)
{
    // Increase L1 cache preference for this kernel
    cudaFuncSetCacheConfig(spmv_csc_kernel_optimized, cudaFuncCachePreferL1);

    const int threads_per_block = 256;
    const int warp_size = 32;
    int warps_per_block = threads_per_block / warp_size;
    int grid_x = (columns + warps_per_block - 1) / warps_per_block;

    dim3 block(threads_per_block, 1, 1);
    dim3 grid(grid_x, 1, 1);

    spmv_csc_kernel_optimized<<<grid, block>>>(
        columns,
        values,
        row_indices,
        col_offsets,
        x,
        y
    );
    cudaDeviceSynchronize();
}
