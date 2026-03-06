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

__global__ void matrix_vector_mul_bf16_kernel(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* x, 
    __nv_bfloat16* y, 
    int M, 
    int N) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            float a = __bfloat162float(A[row * N + col]);
            float b = __bfloat162float(x[col]);
            sum += a * b;
        }
        y[row] = __float2bfloat16(sum);
    }
}

extern "C" void matrix_vector_mul_bf16(
    const __nv_bfloat16* A, 
    const __nv_bfloat16* x, 
    __nv_bfloat16* y, 
    int M, 
    int N) {
    
    const int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    
    matrix_vector_mul_bf16_kernel<<<grid_size, block_size>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}