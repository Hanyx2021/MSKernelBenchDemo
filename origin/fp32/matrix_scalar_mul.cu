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

__global__ void matrix_scalar_mul_kernel(
    const float* A, 
    float* B, 
    float scalar, 
    int M, 
    int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        B[row * N + col] = A[row * N + col] * scalar;
    }
}

extern "C" void matrix_scalar_mul(
    const float* A, 
    float* B, 
    float scalar, 
    int M, 
    int N) {
    
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    matrix_scalar_mul_kernel<<<grid, block>>>(A, B, scalar, M, N);
    cudaDeviceSynchronize();
}