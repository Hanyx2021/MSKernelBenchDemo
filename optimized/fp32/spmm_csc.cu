#include <cuda.h>
#include <cuda_runtime.h>

// Optimized sparse-dense matrix multiplication in CSC format with warp-level broadcasting
constexpr int WARP_SIZE = 32;

__global__ void spmm_csc_kernel_optimized(
    int columns,
    int K,
    const float* values,
    const int* row_indices,
    const int* col_offsets,
    const float* X,
    float* Y)
{
    int k   = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < columns && k < K) {
        float x_val = X[col * K + k];
        int col_start = col_offsets[col];
        int col_end   = col_offsets[col + 1];

        unsigned mask = __activemask();
        int laneId = threadIdx.x & (WARP_SIZE - 1);

        for (int idx = col_start; idx < col_end; ++idx) {
            // Leader lane loads the sparse value and row index
            float v_leader = 0.0f;
            int row_leader = 0;
            if (laneId == 0) {
                v_leader = values[idx];
                row_leader = row_indices[idx];
            }
            // Broadcast to all lanes in the warp
            float v = __shfl_sync(mask, v_leader, 0);
            int row = __shfl_sync(mask, row_leader, 0);
            // Multiply and atomically update
            atomicAdd(&Y[row * K + k], v * x_val);
        }
    }
}

extern "C" void spmm_csc_optimized(
    int rows,
    int columns,
    int K,
    const float* values,
    const int* row_indices,
    const int* col_offsets,
    const float* X,
    float* Y)
{
    // Use blockDim.x as a full warp for warp-level broadcast
    dim3 block(32, 16, 1);
    dim3 grid((K + block.x - 1) / block.x,
              (columns + block.y - 1) / block.y,
              1);

    spmm_csc_kernel_optimized<<<grid, block>>>(
        columns, K,
        values, row_indices, col_offsets,
        X, Y);
    cudaDeviceSynchronize();
}
