#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

__global__ void qkT_bf16_kernel(int q_seq_len, int kv_seq_len, int dim_qk,
                           const __nv_bfloat16* __restrict__ Q,
                           const __nv_bfloat16* __restrict__ K,
                           __nv_bfloat16* __restrict__ S)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= q_seq_len || col >= kv_seq_len) return;
    
    float acc = 0.0f;
    
    for (int k = 0; k < dim_qk; ++k) {
        float a = __bfloat162float(Q[row * dim_qk + k]);
        float b = __bfloat162float(K[col * dim_qk + k]);
        acc += a * b;
    }
    
    float scale = rsqrtf((float)dim_qk);
    S[row * kv_seq_len + col] = __float2bfloat16(acc * scale);
}

__global__ void softmax_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int q_seq_len,
    int kv_seq_len) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < q_seq_len) {
        const __nv_bfloat16* input_row = input + i * kv_seq_len;
        __nv_bfloat16* out_row = out + i * kv_seq_len;

        float maxval = -FLT_MAX;
        for (int j = 0; j < kv_seq_len; j++) {
            float val = __bfloat162float(input_row[j]);
            if (val > maxval) {
                maxval = val;
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < kv_seq_len; j++) {
            float val = __bfloat162float(input_row[j]);
            float exp_val = expf(val - maxval);
            out_row[j] = __float2bfloat16(exp_val);
            sum += exp_val;
        }
        
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < kv_seq_len; j++) {
            float val = __bfloat162float(out_row[j]);
            out_row[j] = __float2bfloat16(val * inv_sum);
        }
    }
}

__global__ void sv_bf16_kernel(int q_seq_len, int kv_seq_len, int dim_v,
                          const __nv_bfloat16* __restrict__ S,
                          const __nv_bfloat16* __restrict__ V,
                          __nv_bfloat16* __restrict__ Y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= q_seq_len || col >= dim_v) return;
    
    float acc = 0.0f;
    
    for (int k = 0; k < kv_seq_len; ++k) {
        float a = __bfloat162float(S[row * kv_seq_len + k]);
        float b = __bfloat162float(V[k * dim_v + col]);
        acc += a * b;
    }
    
    Y[row * dim_v + col] =  __float2bfloat16(acc);
}

extern "C" void softmax_attention_bf16(__nv_bfloat16* Y, const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, 
                                int q_seq_len, int kv_seq_len, int dim_qk, int dim_v)
{
    __nv_bfloat16* S = nullptr;
    __nv_bfloat16* S_softmax = nullptr;
    size_t S_size = (size_t)q_seq_len * (size_t)kv_seq_len * sizeof(__nv_bfloat16);
    cudaMalloc((void**)&S, S_size);
    cudaMalloc((void**)&S_softmax, S_size);

    dim3 block2d(16, 16);

    dim3 grid_qk(
        (kv_seq_len + block2d.x - 1) / block2d.x,
        (q_seq_len + block2d.y - 1) / block2d.y
    );

    qkT_bf16_kernel<<<grid_qk, block2d>>>(q_seq_len, kv_seq_len, dim_qk, Q, K, S);

    int threadsPerBlock = 256;
    int blocks = (q_seq_len + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax_bf16_kernel<<<blocks, threadsPerBlock>>>(S_softmax, S, q_seq_len, kv_seq_len);

    dim3 grid_sv(
        (dim_v + block2d.x - 1) / block2d.x,
        (q_seq_len + block2d.y - 1) / block2d.y
    );
    
    sv_bf16_kernel<<<grid_sv, block2d>>>(q_seq_len, kv_seq_len, dim_v, S_softmax, V, Y);

    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(S_softmax);
}
