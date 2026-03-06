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

__global__ void spmm_coo_kernel(
    int nnz,
    int K,
    const float* values,
    const int* row_indices,
    const int* col_indices,
    const float* X,
    float* Y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < nnz && k < K) {
        float val = values[idx];
        int row = row_indices[idx];
        int col = col_indices[idx];
        
        atomicAdd(&Y[row * K + k], val * X[col * K + k]);
    }
}

extern "C" void spmm_coo(
    int row,
    int nnz,
    int K,
    const float* values,
    const int* row_indices,
    const int* col_indices,
    const float* X,
    float* Y) {
    dim3 block(16, 16, 1);
    dim3 grid((nnz + block.x - 1) / block.x, (K + block.y - 1) / block.y, 1);
    
    spmm_coo_kernel<<<grid, block>>>(nnz, K, values, row_indices, col_indices, X, Y);
    cudaDeviceSynchronize();
}