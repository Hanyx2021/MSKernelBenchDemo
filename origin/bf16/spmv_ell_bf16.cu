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

__global__ void spmv_ell_bf16_kernel(
    int rows,
    int max_nnz_per_row,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *x,
    __nv_bfloat16 *y) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        
        for (int element = 0; element < max_nnz_per_row; element++) {
            int offset = row * max_nnz_per_row + element;
            
            int col = col_ids[offset];
            if (col != -1) {
                float val_f = __bfloat162float(values[offset]);
                float x_val_f = __bfloat162float(x[col]);
                sum += val_f * x_val_f;
            }
        }
        
        y[row] = __float2bfloat16(sum);
    }
}

extern "C" void spmv_ell_bf16(
    int rows,
    int max_nnz_per_row,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *x,
    __nv_bfloat16 *y) {
    dim3 block(256, 1, 1);
    dim3 grid((rows + block.x - 1) / block.x, 1, 1);
    
    spmv_ell_bf16_kernel<<<grid, block>>>(rows, max_nnz_per_row, values, col_ids, x, y);
    cudaDeviceSynchronize();
}