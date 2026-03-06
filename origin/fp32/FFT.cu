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

__global__ void bit_reverse_permute_kernel(
    float* out_real, float* out_img,
    const float* in_real, const float* in_img,
    int N, const int* bit_rev) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    int rev_idx = bit_rev[idx];
    out_real[idx] = in_real[rev_idx];
    out_img[idx] = in_img[rev_idx];
}

__global__ void butterfly_kernel(
    float* data_real, float* data_img,
    const float* twiddle_real, const float* twiddle_img,
    int N, int stage) {
    
    int butterfly_size = 1 << stage;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int num_butterflies = N >> 1;
    
    if (idx >= num_butterflies) return;
    
    int group_id = idx / (butterfly_size >> 1);
    int pair_id = idx % (butterfly_size >> 1);
    
    int idx1 = group_id * butterfly_size + pair_id;
    int idx2 = idx1 + (butterfly_size >> 1);
    
    float x1_real = data_real[idx1];
    float x1_img = data_img[idx1];
    float x2_real = data_real[idx2];
    float x2_img = data_img[idx2];
    
    int twiddle_idx = pair_id * (N >> stage);
    
    float tw_real = twiddle_real[twiddle_idx];
    float tw_img = twiddle_img[twiddle_idx];
    
    float y_real = tw_real * x2_real - tw_img * x2_img;
    float y_img = tw_real * x2_img + tw_img * x2_real;
    
    data_real[idx1] = x1_real + y_real;
    data_img[idx1] = x1_img + y_img;
    data_real[idx2] = x1_real - y_real;
    data_img[idx2] = x1_img - y_img;
}

int* compute_bit_reversal_table_origin(int N) {
    int* table = new int[N];
    int logN = 0;
    for (int i = 1; i < N; i <<= 1)
        logN++;
    
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int j = 0; j < logN; j++) {
            rev = (rev << 1) | ((i >> j) & 1);
        }
        table[i] = rev;
    }
    return table;
}

void compute_twiddle_factors(float* real, float* img, int N) {
    for (int i = 0; i < N/2; i++) {
        float angle = -2.0f * M_PI * i / N;
        double a = angle;
        real[i] = (float)cos(a);
        img[i] = (float)sin(a);
    }
}

extern "C" void FFT(
    const float* input_real, 
    const float* input_img, 
    float* output_real, 
    float* output_img, 
    int N) 
{
    float *d_input_real, *d_input_img;
    float *d_work_real, *d_work_img;
    float *d_twiddle_real, *d_twiddle_img;
    int *d_bit_rev;
    
    size_t mem_size = N * sizeof(float);
    
    cudaMalloc(&d_input_real, mem_size);
    cudaMalloc(&d_input_img, mem_size);
    cudaMalloc(&d_work_real, mem_size);
    cudaMalloc(&d_work_img, mem_size);
    
    cudaMemcpy(d_input_real, input_real, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_img, input_img, mem_size, cudaMemcpyHostToDevice);
    
    int* h_bit_rev = compute_bit_reversal_table_origin(N);
    cudaMalloc(&d_bit_rev, N * sizeof(int));
    cudaMemcpy(d_bit_rev, h_bit_rev, N * sizeof(int), cudaMemcpyHostToDevice);
    
    float* h_twiddle_real = new float[N/2];
    float* h_twiddle_img = new float[N/2];
    compute_twiddle_factors(h_twiddle_real, h_twiddle_img, N);
    
    cudaMalloc(&d_twiddle_real, (N/2) * sizeof(float));
    cudaMalloc(&d_twiddle_img, (N/2) * sizeof(float));
    cudaMemcpy(d_twiddle_real, h_twiddle_real, (N/2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddle_img, h_twiddle_img, (N/2) * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    bit_reverse_permute_kernel<<<blocks, threads_per_block>>>(
        d_work_real, d_work_img, 
        d_input_real, d_input_img, 
        N, d_bit_rev);
    
    cudaDeviceSynchronize();
    
    int stages = 0;
    for (int n = N; n > 1; n >>= 1)
        stages++;
    
    for (int stage = 1; stage <= stages; stage++) {
        int butterflies = N >> 1;
        blocks = (butterflies + threads_per_block - 1) / threads_per_block;
        
        butterfly_kernel<<<blocks, threads_per_block>>>(
            d_work_real, d_work_img,
            d_twiddle_real, d_twiddle_img,
            N, stage);
        
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(output_real, d_work_real, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_img, d_work_img, mem_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input_real);
    cudaFree(d_input_img);
    cudaFree(d_work_real);
    cudaFree(d_work_img);
    cudaFree(d_bit_rev);
    cudaFree(d_twiddle_real);
    cudaFree(d_twiddle_img);
    delete[] h_bit_rev;
    delete[] h_twiddle_real;
    delete[] h_twiddle_img;
}