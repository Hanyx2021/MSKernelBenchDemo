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

__global__ void vector_add_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        C[i] = __float2bfloat16(__bfloat162float(A[i]) + __bfloat162float(B[i]));
    }
}

extern "C" void vector_add_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    vector_add_bf16_kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();
}