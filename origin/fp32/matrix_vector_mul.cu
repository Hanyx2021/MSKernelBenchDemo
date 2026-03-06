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

__global__ void matrix_vector_mul_kernel(
    const float* A, 
    const float* x, 
    float* y, 
    int M, 
    int N) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}

extern "C" void matrix_vector_mul(
    const float* A, 
    const float* x, 
    float* y, 
    int M, 
    int N) {
    
    const int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    
    matrix_vector_mul_kernel<<<grid_size, block_size>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}