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

__global__ void spmm_csc_float_kernel(
    int columns,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_offsets,
    const __nv_bfloat16* X,
    float* Y_float)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < columns && k < K) {
        int col_start = col_offsets[col];
        int col_end = col_offsets[col + 1];
        
        for (int j = col_start; j < col_end; j++) {

            float val_f = __bfloat162float(values[j]);
            float x_val_f = __bfloat162float(X[col * K + k]);
            float result_f = val_f * x_val_f;
        
            int row = row_indices[j];
            atomicAdd(&Y_float[row * K + k], result_f);
        }
    }
}

__global__ void convert_float_to_bf16_kernel(float* float_array, __nv_bfloat16* bf16_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmm_csc_bf16(
    int rows,
    int columns,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y) {

    float* d_Y_float;
    cudaMalloc(&d_Y_float, rows * K * sizeof(float));
    cudaMemset(d_Y_float, 0, rows * K * sizeof(float));

    dim3 block(16, 16, 1);
    dim3 grid((columns + block.x - 1) / block.x, (K + block.y - 1) / block.y, 1);
    
    spmm_csc_float_kernel<<<grid, block>>>(columns, K, values, row_indices, col_offsets, X, d_Y_float);

    dim3 convert_block(256);
    dim3 convert_grid((rows * K + convert_block.x - 1) / convert_block.x);
    convert_float_to_bf16_kernel<<<convert_grid, convert_block>>>(d_Y_float, Y, rows * K);
    
    cudaFree(d_Y_float);
    cudaDeviceSynchronize();
}