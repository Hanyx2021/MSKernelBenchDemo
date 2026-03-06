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


__global__ void square_matrix_mul_kernel(float* A, float* B, float* C, 
    int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < M) {
        float sum = 0.0f;
        for (int k = 0; k < M; k++) {
            sum += A[row * M + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

extern "C" void square_matrix_mul(float* A, float* B, float* C, int M) {
    dim3 block(16, 16, 1);
    dim3 grid((M + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    square_matrix_mul_kernel<<<grid, block>>>(A, B, C, M);
    cudaDeviceSynchronize();
}