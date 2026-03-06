#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <optional>
#include <algorithm>
#include <random>
#include <cmath>
#include <vector>
#include <functional>

__global__ void reverse_array_bf16_kernel(__nv_bfloat16* output, __nv_bfloat16* input, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx > N / 2) return;
    int other_idx = N - 1 - idx;
    __nv_bfloat16 ta = input[idx];
    __nv_bfloat16 tb = input[other_idx];
    output[idx] = tb;
    output[other_idx] = ta;
}

extern "C" void reverse_array_bf16(__nv_bfloat16* output, __nv_bfloat16* input, int N) {
    int threads_needed = (N + 1) / 2;
    int threads_per_block = 256; 
    int blocks = (threads_needed + threads_per_block - 1) / threads_per_block;
    
    reverse_array_bf16_kernel<<<blocks, threads_per_block>>>(output, input, N);
    cudaDeviceSynchronize();
}