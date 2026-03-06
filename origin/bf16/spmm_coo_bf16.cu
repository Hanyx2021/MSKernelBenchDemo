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

__global__ void spmm_coo_float_kernel(
    int nnz,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* X,
    float* Y_float)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < nnz && k < K) {
        float val_f = __bfloat162float(values[idx]);
        int row = row_indices[idx];
        int col = col_indices[idx];

        float x_val_f = __bfloat162float(X[col * K + k]);
        float result_f = val_f * x_val_f;
        
        atomicAdd(&Y_float[row * K + k], result_f);
    }
}

__global__ void convert_float_to_bf16_kernel(float* float_array, __nv_bfloat16* bf16_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmm_coo_bf16(
    int rows,
    int nnz,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y) {
    
    float* d_Y_float;
    cudaMalloc(&d_Y_float, rows * K * sizeof(float));
    cudaMemset(d_Y_float, 0, rows * K * sizeof(float));
    
    dim3 block(16, 16, 1);
    dim3 grid((nnz + block.x - 1) / block.x, (K + block.y - 1) / block.y, 1);
    
    spmm_coo_float_kernel<<<grid, block>>>(nnz, K, values, row_indices, col_indices, X, d_Y_float);
    
    dim3 convert_block(256);
    dim3 convert_grid((rows * K + convert_block.x - 1) / convert_block.x);
    convert_float_to_bf16_kernel<<<convert_grid, convert_block>>>(d_Y_float, Y, rows * K);
    
    cudaFree(d_Y_float);
    cudaDeviceSynchronize();
}