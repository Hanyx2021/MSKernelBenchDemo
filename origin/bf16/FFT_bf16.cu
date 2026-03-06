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

__global__ void bit_reverse_permute_bf16_kernel(
    __nv_bfloat16* out_real, __nv_bfloat16* out_img,
    const __nv_bfloat16* in_real, const __nv_bfloat16* in_img,
    int N, const int* bit_rev) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    int rev_idx = bit_rev[idx];
    out_real[idx] = in_real[rev_idx];
    out_img[idx] = in_img[rev_idx];
}

__global__ void butterfly_bf16_kernel(
    __nv_bfloat16* data_real, __nv_bfloat16* data_img,
    const __nv_bfloat16* twiddle_real, const __nv_bfloat16* twiddle_img,
    int N, int stage) {
    
    int butterfly_size = 1 << stage;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int num_butterflies = N >> 1;
    
    if (idx >= num_butterflies) return;
    
    int group_id = idx / (butterfly_size >> 1);
    int pair_id = idx % (butterfly_size >> 1);
    
    int idx1 = group_id * butterfly_size + pair_id;
    int idx2 = idx1 + (butterfly_size >> 1);
    
    float x1_real = __bfloat162float(data_real[idx1]);
    float x1_img = __bfloat162float(data_img[idx1]);
    float x2_real = __bfloat162float(data_real[idx2]);
    float x2_img = __bfloat162float(data_img[idx2]);
    
    int twiddle_idx = pair_id * (N >> stage);
    
    float tw_real = __bfloat162float(twiddle_real[twiddle_idx]);
    float tw_img = __bfloat162float(twiddle_img[twiddle_idx]);
    
    float y_real = tw_real * x2_real - tw_img * x2_img;
    float y_img = tw_real * x2_img + tw_img * x2_real;
    
    data_real[idx1] = __float2bfloat16(x1_real + y_real);
    data_img[idx1] = __float2bfloat16(x1_img + y_img);
    data_real[idx2] = __float2bfloat16(x1_real - y_real);
    data_img[idx2] = __float2bfloat16(x1_img - y_img);
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

void compute_twiddle_factors_bf16(__nv_bfloat16* real, __nv_bfloat16* img, int N) {
    for (int i = 0; i < N/2; i++) {
        float angle = -2.0f * M_PI * i / N;
        double a = angle;
        real[i] = __float2bfloat16(cos(a));
        img[i] = __float2bfloat16(sin(a));
    }
}

extern "C" void FFT_bf16(
    const __nv_bfloat16* input_real, 
    const __nv_bfloat16* input_img, 
    __nv_bfloat16* output_real, 
    __nv_bfloat16* output_img, 
    int N) 
{
    __nv_bfloat16 *d_input_real, *d_input_img;
    __nv_bfloat16 *d_work_real, *d_work_img;
    __nv_bfloat16 *d_twiddle_real, *d_twiddle_img;
    int *d_bit_rev;
    
    size_t mem_size = N * sizeof(__nv_bfloat16);
    
    cudaMalloc(&d_input_real, mem_size);
    cudaMalloc(&d_input_img, mem_size);
    cudaMalloc(&d_work_real, mem_size);
    cudaMalloc(&d_work_img, mem_size);
    
    cudaMemcpy(d_input_real, input_real, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_img, input_img, mem_size, cudaMemcpyHostToDevice);
    
    int* h_bit_rev = compute_bit_reversal_table_origin(N);
    cudaMalloc(&d_bit_rev, N * sizeof(int));
    cudaMemcpy(d_bit_rev, h_bit_rev, N * sizeof(int), cudaMemcpyHostToDevice);
    
    __nv_bfloat16* h_twiddle_real = new __nv_bfloat16[N/2];
    __nv_bfloat16* h_twiddle_img = new __nv_bfloat16[N/2];
    compute_twiddle_factors_bf16(h_twiddle_real, h_twiddle_img, N);
    
    cudaMalloc(&d_twiddle_real, (N/2) * sizeof(__nv_bfloat16));
    cudaMalloc(&d_twiddle_img, (N/2) * sizeof(__nv_bfloat16));
    cudaMemcpy(d_twiddle_real, h_twiddle_real, (N/2) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddle_img, h_twiddle_img, (N/2) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    bit_reverse_permute_bf16_kernel<<<blocks, threads_per_block>>>(
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
        
        butterfly_bf16_kernel<<<blocks, threads_per_block>>>(
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