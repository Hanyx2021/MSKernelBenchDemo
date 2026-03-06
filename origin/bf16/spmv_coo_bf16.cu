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

__global__ void spmv_coo_float_kernel(
    int nnz,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* x,
    float* y_float)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        float val_f = __bfloat162float(values[idx]);
        int row = row_indices[idx];
        int col = col_indices[idx];

        float x_val_f = __bfloat162float(x[col]);
        float result_f = val_f * x_val_f;
        
        atomicAdd(&y_float[row], result_f);
    }
}

__global__ void convert_float_to_bf16_kernel(float* float_array, __nv_bfloat16* bf16_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmv_coo_bf16(
    int rows,
    int nnz,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* x,
    __nv_bfloat16* y) {
    
    float* d_y_float;
    cudaMalloc(&d_y_float, rows * sizeof(float));
    cudaMemset(d_y_float, 0, rows * sizeof(float));
    
    dim3 block(256, 1, 1);
    dim3 grid((nnz + block.x - 1) / block.x, 1, 1);
    
    spmv_coo_float_kernel<<<grid, block>>>(
        nnz, values, row_indices, col_indices, x, d_y_float);
    
    dim3 convert_block(256);
    dim3 convert_grid((rows + convert_block.x - 1) / convert_block.x);
    convert_float_to_bf16_kernel<<<convert_grid, convert_block>>>(d_y_float, y, rows);
    
    cudaFree(d_y_float);
    cudaDeviceSynchronize();
}