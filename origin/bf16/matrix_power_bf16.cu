#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

__global__ void matrix_mul_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            float a_val = __bfloat162float(A[row * N + k]);
            float b_val = __bfloat162float(B[k * N + col]);
            sum += a_val * b_val;
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

extern "C" void matrix_power_bf16(
    const __nv_bfloat16* A, 
    __nv_bfloat16* B, 
    int N, 
    int P) {

    __nv_bfloat16* d_A;
    __nv_bfloat16* d_B;
    __nv_bfloat16* d_temp;
    
    cudaMalloc(&d_A, N * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, N * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_temp, N * N * sizeof(__nv_bfloat16));
    
    cudaMemcpy(d_A, A, N * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
    
    matrix_mul_bf16_kernel<<<grid, block>>>(d_A, d_A, d_B, N);
    
    for (int i = 2; i < P; i++) {
        matrix_mul_bf16_kernel<<<grid, block>>>(d_B, d_A, d_temp, N);
        
        cudaMemcpy(d_B, d_temp, N * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    }
    
    cudaMemcpy(B, d_B, N * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_temp);
}