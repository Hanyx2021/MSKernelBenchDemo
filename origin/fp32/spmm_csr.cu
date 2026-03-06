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


__global__ void spmm_csr_kernel(
    int rows,
    int K,
    const float* values,
    const int* col_indices,
    const int* row_offsets,
    const float* X,
    float* Y)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_k = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col_k < K) {
        float sum = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];

        for (int j = row_start; j < row_end; j++) {
            int col_index = col_indices[j];
            sum += values[j] * X[col_index * K + col_k];
        }

        Y[row * K + col_k] = sum;
    }
}

extern "C" void spmm_csr(
    int rows,
    int K,
    const float* values,
    const int* col_indices,
    const int* row_offsets,
    const float* X,
    float* Y) {
    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (rows + 15) / 16);
    
    spmm_csr_kernel<<<grid, block>>>(rows, K, values, col_indices, row_offsets, X, Y);
    cudaDeviceSynchronize();
}