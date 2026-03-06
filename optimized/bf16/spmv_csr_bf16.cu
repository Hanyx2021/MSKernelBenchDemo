#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Warp-cooperative CSR SpMV kernel with bfloat16 inputs/outputs
__global__ void spmv_csr_bf16_kernel_optimized(
    int rows,
    const __nv_bfloat16* __restrict__ values,
    const int* __restrict__ col_indices,
    const int* __restrict__ row_offsets,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y)
{
    // Identify warp and lane
    int warpId = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int laneId = threadIdx.x & 31;
    if (warpId >= rows) return;
    int row = warpId;

    int row_start = row_offsets[row];
    int row_end   = row_offsets[row + 1];
    float partial = 0.0f;

    // Each lane processes a strided subset of non-zeros in the row
    for (int j = row_start + laneId; j < row_end; j += 32) {
        float v = __bfloat162float(values[j]);
        int idx = col_indices[j];
        // Load x through read-only data cache
        float xv = __bfloat162float(__ldg(x + idx));
        partial += v * xv;
    }

    // Intra-warp reduction to accumulate partial sums
    unsigned int fullMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(fullMask, partial, offset);
    }

    // Write result by lane 0 of each warp
    if (laneId == 0) {
        y[row] = __float2bfloat16(partial);
    }
}

extern "C" void spmv_csr_bf16_optimized(
    int rows,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* x,
    __nv_bfloat16* y)
{
    // Warp-cooperative setup: choose block of multiple warps
    const int warpSize = 32;
    const int warpsPerBlock = 8;                // Tuneable: e.g., 8 warps per block => 256 threads
    const int blockSize = warpsPerBlock * warpSize;
    dim3 block(blockSize, 1, 1);
    // Each warp handles one row, so grid size = ceil(rows / warpsPerBlock)
    dim3 grid((rows + warpsPerBlock - 1) / warpsPerBlock, 1, 1);

    spmv_csr_bf16_kernel_optimized<<<grid, block>>>(
        rows, values, col_indices, row_offsets, x, y);
    cudaDeviceSynchronize();
}
