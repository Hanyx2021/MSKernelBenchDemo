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

__global__ void leaky_RELU_kernel(
    float* output,
    const float* input,
    const int N,
    float alpha = 0.01) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if(input[i] < 0)
            output[i] = alpha * input[i];
        else
            output[i] = input[i];
    }
}

extern "C" void leaky_RELU(
    float* output,
    const float* input,
    const int N,
    float alpha = 0.01) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    leaky_RELU_kernel<<<grid, block>>>(output, input, N, alpha);
    cudaDeviceSynchronize();
}