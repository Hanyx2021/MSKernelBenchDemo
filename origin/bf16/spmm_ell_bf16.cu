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

__global__ void spmm_ell_bf16_kernel(
    int rows,
    int max_nnz_per_row,
    int K,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *X,
    __nv_bfloat16 *Y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && k < K) {
        float sum = 0.0f;
        
        for (int element = 0; element < max_nnz_per_row; element++) {
            int offset = row * max_nnz_per_row + element;
            int col = col_ids[offset];
            
            if (col != -1) {
                float val_f = __bfloat162float(values[offset]);
                float x_val_f = __bfloat162float(X[col * K + k]);
                sum += val_f * x_val_f;
            }
        }
        
        Y[row * K + k] = __float2bfloat16(sum);
    }
}

extern "C" void spmm_ell_bf16(
    int rows,
    int max_nnz_per_row,
    int K,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *X,
    __nv_bfloat16 *Y) {
    dim3 block(16, 16, 1);
    dim3 grid((rows + block.x - 1) / block.x, (K + block.y - 1) / block.y, 1);
    
    spmm_ell_bf16_kernel<<<grid, block>>>(rows, max_nnz_per_row, K, values, col_ids, X, Y);
    cudaDeviceSynchronize();
}