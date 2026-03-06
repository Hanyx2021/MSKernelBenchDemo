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

__global__ void spmv_csc_kernel(
    int columns,
    const float* values,
    const int* row_indices,
    const int* col_offsets,
    const float* x,
    float* y)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < columns) {
        float x_val = x[col];
        int col_start = col_offsets[col];
        int col_end = col_offsets[col + 1];
        
        for (int j = col_start; j < col_end; j++) {
            int row = row_indices[j];
            float val = values[j];

            atomicAdd(&y[row], val * x_val);
        }
    }
}

extern "C" void spmv_csc(
    int rows,
    int columns,
    const float* values,
    const int* row_indices,
    const int* col_offsets,
    const float* x,
    float* y) {
    dim3 block(256, 1, 1);
    dim3 grid((columns + block.x - 1) / block.x, 1, 1);
    
    spmv_csc_kernel<<<grid, block>>>(columns, values, row_indices, col_offsets, x, y);
    cudaDeviceSynchronize();
}