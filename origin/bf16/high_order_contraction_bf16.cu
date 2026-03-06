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
#include <cub/cub.cuh>

__global__ void high_order_contraction_bf16_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_c_elements = a_dim * b_dim * c_dim;
    
    if (idx >= total_c_elements) {
        return;
    }

    int c = idx % c_dim;
    idx = idx / c_dim;
    int b = idx % b_dim;
    int a = idx / b_dim;
    
    float sum = 0.0f;

    for (int x = 0; x < x_dim; x++) {
        for (int y = 0; y < y_dim; y++) {

            int idx_A = ((a * x_dim + x) * b_dim + b) * y_dim + y;
   
            int idx_B = (x * c_dim + c) * y_dim + y;

            sum += __bfloat162float(A[idx_A]) * __bfloat162float(B[idx_B]);
        }
    }
    
    int idx_C = (a * b_dim + b) * c_dim + c;
    
    C[idx_C] = __float2bfloat16(sum);
}

extern "C" void high_order_contraction_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim) {

    size_t total_c_elements = a_dim * b_dim * c_dim;
    int block_size = 256;
    int grid_size = (total_c_elements + block_size - 1) / block_size;

    high_order_contraction_bf16_kernel<<<grid_size, block_size>>>(
        A, B, C,
        a_dim, b_dim, c_dim,
        x_dim, y_dim
    );
}