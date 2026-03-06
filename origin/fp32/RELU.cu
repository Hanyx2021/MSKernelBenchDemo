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

__global__ void RELU_kernel(
    float* output,
    const float* input,
    const int N) {
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = max(0.0f, input[i]);
    }
}

extern "C" void RELU(
    float* output,
    const float* input,
    int N) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    RELU_kernel<<<grid, block>>>(output, input, N);
    cudaDeviceSynchronize();
}