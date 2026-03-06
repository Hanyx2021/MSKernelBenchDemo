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

__global__ void matrix_mul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" void matrix_power(
    const float* A, 
    float* B, 
    int N, 
    int P) {

    float* d_A;
    float* d_B;
    float* d_temp;
    
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_temp, N * N * sizeof(float));
    
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
    
    matrix_mul_kernel<<<grid, block>>>(d_A, d_A, d_B, N);
    
    for (int i = 2; i < P; i++) {
        matrix_mul_kernel<<<grid, block>>>(d_B, d_A, d_temp, N);
        
        cudaMemcpy(d_B, d_temp, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    cudaMemcpy(B, d_B, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_temp);
}