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

__global__ void matrix_scalar_mul_bf16_kernel(
    const __nv_bfloat16* A, 
    __nv_bfloat16* B, 
    __nv_bfloat16 scalar, 
    int M, 
    int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float val = __bfloat162float(A[row * N + col]);
        float scaled = val * __bfloat162float(scalar);
        B[row * N + col] = __float2bfloat16(scaled);
    }
}

extern "C" void matrix_scalar_mul_bf16(
    const __nv_bfloat16* A, 
    __nv_bfloat16* B, 
    __nv_bfloat16 scalar, 
    int M, 
    int N) {
    
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    matrix_scalar_mul_bf16_kernel<<<grid, block>>>(A, B, scalar, M, N);
    cudaDeviceSynchronize();
}