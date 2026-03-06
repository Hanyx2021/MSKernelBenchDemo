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
#include <float.h>


__global__ void softmax_kernel(
    float* out,
    const float* input,
    int N,
    int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* input_row = input + i * C;
        float* out_row = out + i * C;

        float maxval = -FLT_MAX;
        for (int j = 0; j < C; j++) {
            if (input_row[j] > maxval) {
                maxval = input_row[j];
            }
        }
        float sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(input_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;
        }
    }
}

extern "C" void softmax(
    float* out,
    const float* input,
    int N,
    int C) {
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    softmax_kernel<<<grid, block>>>(out, input, N, C);
    cudaDeviceSynchronize();
}