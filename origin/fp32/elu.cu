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

__global__ void elu_kernel(
    float* output,
    const float* input,
    const int N,
    float alpha = 1.0) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if(input[i] < 0)
            output[i] = alpha * (exp(input[i]) - 1);
        else
            output[i] = input[i];
    }
}

extern "C" void elu(
    float* output,
    const float* input,
    const int N,
    float alpha = 1.0) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    elu_kernel<<<grid, block>>>(output, input, N, alpha);
    cudaDeviceSynchronize();
}