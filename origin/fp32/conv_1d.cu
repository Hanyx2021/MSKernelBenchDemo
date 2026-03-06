#include <cuda.h>
#include <cuda_runtime.h>
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

__global__ void conv_1d_kernel(const float* input, const float* kernel, float* output,
                             int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= input_size - kernel_size + 1) return;
    float temp = 0.0f;
    for(int i = 0; i < kernel_size; i++){
        float tempI = input[idx + i];
        float tempK = kernel[i];
        temp += tempI * tempK;
    }
    output[idx] = temp;
}

extern "C" void conv_1d(const float* input, const float* kernel, float* output,
                       int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock, 1, 1);
    dim3 grid(blocksPerGrid, 1, 1);
    
    conv_1d_kernel<<<grid, block>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}