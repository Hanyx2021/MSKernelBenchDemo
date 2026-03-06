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
#include <tuple>
#include <float.h>
#include <functional>


__global__ void square_matrix_mul_bf16_kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, 
    int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < M) {
        float sum = 0.0f;
        for (int k = 0; k < M; k++) {
            float a_val = __bfloat162float(A[row * M + k]);
            float b_val = __bfloat162float(B[k * M + col]);
            sum += a_val * b_val;
        }
        C[row * M + col] = __float2bfloat16(sum);
    }
}

extern "C" void square_matrix_mul_bf16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M) {
    dim3 block(16, 16, 1);
    dim3 grid((M + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);
    
    square_matrix_mul_bf16_kernel<<<grid, block>>>(A, B, C, M);
    cudaDeviceSynchronize();
}