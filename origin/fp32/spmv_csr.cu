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

__global__ void spmv_csr_kernel(
    int rows,
    const float* values,
    const int* col_indices,
    const int* row_offsets,
    const float* x,
    float* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        
        y[row] = sum;
    }
}

extern "C" void spmv_csr(
    int rows,
    const float* values,
    const int* col_indices,
    const int* row_offsets,
    const float* x,
    float* y) {
    dim3 block(32, 32, 1);
    dim3 grid((rows + 31) / 32, 1, 1);
    
    spmv_csr_kernel<<<grid, block>>>(rows, values, col_indices, row_offsets, x, y);
    cudaDeviceSynchronize();
}