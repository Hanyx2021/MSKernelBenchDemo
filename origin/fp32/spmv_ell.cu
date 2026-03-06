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

__global__ void spmv_ell_kernel(
    int rows,
    int max_nnz_per_row,
    const float *values,
    const int *col_ids,
    const float *x,
    float *y) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        
        for (int element = 0; element < max_nnz_per_row; element++) {
            int offset = row * max_nnz_per_row + element;
            
            int col = col_ids[offset];
            if (col != -1) {
                float val = values[offset];
                sum += val * x[col];
            }
        }
        
        y[row] = sum;
    }
}

extern "C" void spmv_ell(
    int rows,
    int max_nnz_per_row,
    const float *values,
    const int *col_ids,
    const float *x,
    float *y) {
    dim3 block(256, 1, 1);
    dim3 grid((rows + block.x - 1) / block.x, 1, 1);
    
    spmv_ell_kernel<<<grid, block>>>(rows, max_nnz_per_row, values, col_ids, x, y);
    cudaDeviceSynchronize();
}