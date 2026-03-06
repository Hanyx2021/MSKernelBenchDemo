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


__global__ void spmm_ell_kernel(
    int rows,
    int max_nnz_per_row,
    int K,
    const float *values,
    const int *col_ids,
    const float *X,
    float *Y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && k < K) {
        float sum = 0.0f;
        
        for (int element = 0; element < max_nnz_per_row; element++) {
            int offset = row * max_nnz_per_row + element;
            int col = col_ids[offset];
            
            if (col != -1) {
                float val = values[offset];
                sum += val * X[col * K + k];
            }
        }
        
        Y[row * K + k] = sum;
    }
}

extern "C" void spmm_ell(
    int rows,
    int max_nnz_per_row,
    int K,
    const float *values,
    const int *col_ids,
    const float *X,
    float *Y) {
    dim3 block(16, 16, 1);
    dim3 grid((rows + block.x - 1) / block.x, (K + block.y - 1) / block.y, 1);
    
    spmm_ell_kernel<<<grid, block>>>(rows, max_nnz_per_row, K, values, col_ids, X, Y);
    cudaDeviceSynchronize();
}