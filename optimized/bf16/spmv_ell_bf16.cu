#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Optimized ELL-format SpMV kernel using warp-striping and bfloat16 values
__global__ void spmv_ell_bf16_kernel_optimized(
    int rows,
    int max_nnz_per_row,
    const __nv_bfloat16 *__restrict__ values,
    const int *__restrict__ col_ids,
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ y) {
    // Compute warp and lane identifiers
    int lane_id = threadIdx.x & 31;
    int warp_id_in_block = threadIdx.x >> 5;               // threadIdx.x / 32
    int warps_per_block = blockDim.x >> 5;                 // blockDim.x / 32
    int warp_global_id = blockIdx.x * warps_per_block + warp_id_in_block;

    // Each warp works on one row
    if (warp_global_id >= rows) return;
    int row = warp_global_id;

    float lane_sum = 0.0f;
    // Warp-striping over non-zero entries for full coalescing
    for (int idx = lane_id; idx < max_nnz_per_row; idx += 32) {
        int offset = row * max_nnz_per_row + idx;
        // Load value and column index via read-only cache
        __nv_bfloat16 bf16_v = __ldg(&values[offset]);
        int c = __ldg(&col_ids[offset]);
        if (c >= 0) {
            float v_f = __bfloat162float(bf16_v);
            float x_f = __bfloat162float(__ldg(&x[c]));
            lane_sum += v_f * x_f;
        }
    }

    // Warp-synchronous reduction of lane sums
    for (int s = 16; s > 0; s >>= 1) {
        lane_sum += __shfl_down_sync(0xFFFFFFFF, lane_sum, s);
    }

    // Write result by lane 0 of each warp
    if (lane_id == 0) {
        y[row] = __float2bfloat16(lane_sum);
    }
}

extern "C" void spmv_ell_bf16_optimized(
    int rows,
    int max_nnz_per_row,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *x,
    __nv_bfloat16 *y) {
    // Configure launch: 256 threads per block (8 warps)
    const int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    dim3 block(threads_per_block, 1, 1);
    dim3 grid((rows + warps_per_block - 1) / warps_per_block, 1, 1);

    spmv_ell_bf16_kernel_optimized<<<grid, block>>>(
        rows, max_nnz_per_row, values, col_ids, x, y);
    cudaDeviceSynchronize();
}