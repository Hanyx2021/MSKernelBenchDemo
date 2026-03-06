#include <cuda.h>
#include <cuda_runtime.h>
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


__global__ void spmm_csc_kernel(
    int columns,
    int K,
    const float* values,
    const int* row_indices,
    const int* col_offsets,
    const float* X,
    float* Y)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < columns && k < K) {
        int col_start = col_offsets[col];
        int col_end = col_offsets[col + 1];
        
        for (int j = col_start; j < col_end; j++) {
            int row = row_indices[j];
            float val = values[j];
            atomicAdd(&Y[row * K + k], val * X[col * K + k]);
        }
    }
}

extern "C" void spmm_csc(
    int rows,
    int columns,
    int K,
    const float* values,
    const int* row_indices,
    const int* col_offsets,
    const float* X,
    float* Y) {
    dim3 block(16, 16, 1);
    dim3 grid((columns + block.x - 1) / block.x, (K + block.y - 1) / block.y, 1);
    
    spmm_csc_kernel<<<grid, block>>>(columns, K, values, row_indices, col_offsets, X, Y);
    cudaDeviceSynchronize();
}