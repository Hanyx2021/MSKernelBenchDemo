#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <optional>
#include <algorithm>
#include <random>
#include <cmath>
#include <vector>
#include <functional>

__global__ void spmv_csr_bf16_kernel(
    int rows,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* x,
    __nv_bfloat16* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];
        
        for (int j = row_start; j < row_end; j++) {
            float val_f = __bfloat162float(values[j]);
            float x_val_f = __bfloat162float(x[col_indices[j]]);
            sum += val_f * x_val_f;
        }
        
        y[row] = __float2bfloat16(sum);
    }
}

extern "C" void spmv_csr_bf16(
    int rows,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* x,
    __nv_bfloat16* y) {
    dim3 block(32, 32, 1);
    dim3 grid((rows + 31) / 32, 1, 1);
    
    spmv_csr_bf16_kernel<<<grid, block>>>(rows, values, col_indices, row_offsets, x, y);
    cudaDeviceSynchronize();
}