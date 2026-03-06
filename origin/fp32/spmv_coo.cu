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

__global__ void spmv_coo_kernel(
    int nnz,
    const float* values,
    const int* row_indices,
    const int* col_indices,
    const float* x,
    float* y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        float val = values[idx];
        int row = row_indices[idx];
        int col = col_indices[idx];

        atomicAdd(&y[row], val * x[col]);
    }
}

extern "C" void spmv_coo(
    int rows,
    int nnz,
    const float* values,
    const int* row_indices,
    const int* col_indices,
    const float* x,
    float* y) {
    dim3 block(256, 1, 1);
    dim3 grid((nnz + block.x - 1) / block.x, 1, 1);
    
    spmv_coo_kernel<<<grid, block>>>(nnz, values, row_indices, col_indices, x, y);
    cudaDeviceSynchronize();
}