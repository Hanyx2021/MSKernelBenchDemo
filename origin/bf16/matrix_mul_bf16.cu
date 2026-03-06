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

__global__ void matrix_mul_bf16_kernel(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    __nv_bfloat16* C, 
    int M, 
    int N, 
    int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = __bfloat162float(A[row * K + k]);
            float b = __bfloat162float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

extern "C" void matrix_mul_bf16(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* B, 
    __nv_bfloat16* C, 
    int M, 
    int N, 
    int K) {
    
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    matrix_mul_bf16_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}