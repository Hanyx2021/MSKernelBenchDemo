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

__global__ void qkT_kernel(int q_seq_len, int kv_seq_len, int dim_qk,
                           const float* __restrict__ Q,
                           const float* __restrict__ K,
                           float* __restrict__ S)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= q_seq_len || col >= kv_seq_len) return;
    
    float acc = 0.0f;
    
    for (int k = 0; k < dim_qk; ++k) {
        acc += Q[row * dim_qk + k] * K[col * dim_qk + k];
    }
    
    float scale = rsqrtf((float)dim_qk);
    S[row * kv_seq_len + col] = acc * scale;
}

__global__ void softmax_kernel(
    float* out,
    const float* input,
    int q_seq_len,
    int kv_seq_len) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < q_seq_len) {
        const float* input_row = input + i * kv_seq_len;
        float* out_row = out + i * kv_seq_len;

        float maxval = -FLT_MAX;
        for (int j = 0; j < kv_seq_len; j++) {
            if (input_row[j] > maxval) {
                maxval = input_row[j];
            }
        }
        float sum = 0.0;
        for (int j = 0; j < kv_seq_len; j++) {
            out_row[j] = expf(input_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < kv_seq_len; j++) {
            out_row[j] /= (float)sum;
        }
    }
}

__global__ void sv_kernel(int q_seq_len, int kv_seq_len, int dim_v,
                          const float* __restrict__ S,
                          const float* __restrict__ V,
                          float* __restrict__ Y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= q_seq_len || col >= dim_v) return;
    
    float acc = 0.0f;
    
    for (int k = 0; k < kv_seq_len; ++k) {
        acc += S[row * kv_seq_len + k] * V[k * dim_v + col];
    }
    
    Y[row * dim_v + col] = acc;
}

extern "C" void softmax_attention(float* Y, const float* Q, const float* K, const float* V, 
                                int q_seq_len, int kv_seq_len, int dim_qk, int dim_v)
{
    float* S = nullptr;
    float* S_softmax = nullptr;
    size_t S_size = (size_t)q_seq_len * (size_t)kv_seq_len * sizeof(float);
    cudaMalloc((void**)&S, S_size);
    cudaMalloc((void**)&S_softmax, S_size);

    dim3 block2d(16, 16);

    dim3 grid_qk(
        (kv_seq_len + block2d.x - 1) / block2d.x,
        (q_seq_len + block2d.y - 1) / block2d.y
    );

    qkT_kernel<<<grid_qk, block2d>>>(q_seq_len, kv_seq_len, dim_qk, Q, K, S);

    int threadsPerBlock = 256;
    int blocks = (q_seq_len + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax_kernel<<<blocks, threadsPerBlock>>>(S_softmax, S, q_seq_len, kv_seq_len);

    dim3 grid_sv(
        (dim_v + block2d.x - 1) / block2d.x,
        (q_seq_len + block2d.y - 1) / block2d.y
    );
    
    sv_kernel<<<grid_sv, block2d>>>(q_seq_len, kv_seq_len, dim_v, S_softmax, V, Y);

    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(S_softmax);
}
