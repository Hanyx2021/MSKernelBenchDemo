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

__global__ void matrix_copy_bf16_kernel(__nv_bfloat16* A, __nv_bfloat16* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        B[row * N + col] = A[row * N + col];
    }
}

extern "C" void matrix_copy_bf16(__nv_bfloat16* A, __nv_bfloat16* B, int M, int N) {
    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    matrix_copy_bf16_kernel<<<grid, block>>>(A, B, M, N);
    cudaDeviceSynchronize();
}