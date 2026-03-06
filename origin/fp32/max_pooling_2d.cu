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

__global__ void max_pooling_2d_kernel(
    const float* input,
    float* output,
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_H,
    const int out_W,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h_out >= out_H || w_out >= out_W || n >= N || c >= C) {
        return;
    }
    
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    int h_end = min(h_start + kernel_size, H + padding);
    int w_end = min(w_start + kernel_size, W + padding);
    
    float max_val = -FLT_MAX;
    
    for (int h = max(h_start, 0); h < min(h_end, H); h++) {
        for (int w = max(w_start, 0); w < min(w_end, W); w++) {
            int input_idx = ((n * C + c) * H + h) * W + w;
            float val = input[input_idx];
            
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    
    int output_idx = ((n * C + c) * out_H + h_out) * out_W + w_out;
    output[output_idx] = max_val;
}

extern "C" void max_pooling_2d(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int kernel_size,
    int stride,
    int padding) {
    
    int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    int out_W = (W + 2 * padding - kernel_size) / stride + 1;
    
    if (out_H <= 0 || out_W <= 0) {
        return;
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (out_W + blockSize.x - 1) / blockSize.x,
        (out_H + blockSize.y - 1) / blockSize.y,
        N * C
    );
    
    max_pooling_2d_kernel<<<gridSize, blockSize>>>(
        input, output, N, C, H, W, out_H, out_W, kernel_size, stride, padding);
    
    cudaDeviceSynchronize();
}