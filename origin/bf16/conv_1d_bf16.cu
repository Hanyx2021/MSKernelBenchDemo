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

__global__ void conv_1d_bf16_kernel(const __nv_bfloat16* input, const __nv_bfloat16* kernel, __nv_bfloat16* output,
                                   int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= input_size - kernel_size + 1) return;
    float temp = 0.0f;
    for(int i = 0; i < kernel_size; i++){
        __nv_bfloat16 tempI = input[idx + i];
        __nv_bfloat16 tempK = kernel[i];
        temp += __bfloat162float(tempI) * __bfloat162float(tempK);
    }
    output[idx] = __float2bfloat16(temp);
}

extern "C" void conv_1d_bf16(const __nv_bfloat16* input, const __nv_bfloat16* kernel, __nv_bfloat16* output,
                           int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock, 1, 1);
    dim3 grid(blocksPerGrid, 1, 1);
    
    conv_1d_bf16_kernel<<<grid, block>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}