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

__global__ void selu_kernel(
    float* output,
    const float* input,
    const int N,
    float alpha = 1.67,
    float lambda = 1.0) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if(input[i] < 0)
            output[i] = lambda * alpha * (exp(input[i]) - 1);
        else
            output[i] = lambda * input[i];
    }
}

extern "C" void selu(
    float* output,
    const float* input,
    const int N,
    float alpha = 1.67,
    float lambda = 1.0) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    selu_kernel<<<grid, block>>>(output, input, N, alpha, lambda);
    cudaDeviceSynchronize();
}