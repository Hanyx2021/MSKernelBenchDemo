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
#include <cub/cub.cuh>

__global__ void triangle_attention_bf16_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float* attention_mask,
    __nv_bfloat16* output,
    int batch_size,
    int head_num,
    int seq_length,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * head_num * seq_length * head_dim;
    
    if (idx >= total_elements) return;

    int b = idx / (head_num * seq_length * head_dim);
    int remaining = idx % (head_num * seq_length * head_dim);
    int h = remaining / (seq_length * head_dim);
    remaining = remaining % (seq_length * head_dim);
    int pos_i = remaining / head_dim;
    int d = remaining % head_dim;
    
    float sum = 0.0f;
    float max_val = -FLT_MAX;

    for (int pos_j = 0; pos_j <= pos_i; pos_j++) {
        float q_val = __bfloat162float(query[idx]);
        int k_idx = b * head_num * seq_length * head_dim + h * seq_length * head_dim + pos_j * head_dim + d;
        float k_val = __bfloat162float(key[k_idx]);
        
        float score = q_val * k_val / sqrtf((float)head_dim);
        
        int mask_idx = b * seq_length * seq_length + pos_i * seq_length + pos_j;
        float mask_val = attention_mask[mask_idx];
        
        if (mask_val > 0.5f) {
            score = -FLT_MAX;
        }
        
        if (score > max_val) max_val = score;
    }
    
    float sum_exp = 0.0f;
    for (int pos_j = 0; pos_j <= pos_i; pos_j++) {
        float q_val = __bfloat162float(query[idx]);
        int k_idx = b * head_num * seq_length * head_dim + h * seq_length * head_dim + pos_j * head_dim + d;
        float k_val = __bfloat162float(key[k_idx]);
        
        float score = q_val * k_val / sqrtf((float)head_dim);
        
        int mask_idx = b * seq_length * seq_length + pos_i * seq_length + pos_j;
        float mask_val = attention_mask[mask_idx];
        
        if (mask_val > 0.5f) {
            score = -FLT_MAX;
        }
        
        float exp_score;
        if (score <= -FLT_MAX) {
            exp_score = 0.0f;
        } else {
            exp_score = expf(score - max_val);
        }
        sum_exp += exp_score;
    }
    
    if (sum_exp > 0.0f) {
        for (int pos_j = 0; pos_j <= pos_i; pos_j++) {
            float q_val = __bfloat162float(query[idx]);
            int k_idx = b * head_num * seq_length * head_dim + h * seq_length * head_dim + pos_j * head_dim + d;
            float k_val = __bfloat162float(key[k_idx]);
            
            float score = q_val * k_val / sqrtf((float)head_dim);
            
            int mask_idx = b * seq_length * seq_length + pos_i * seq_length + pos_j;
            float mask_val = attention_mask[mask_idx];
            
            if (mask_val > 0.5f) {
                score = -FLT_MAX;
            }
            
            float exp_score;
            if (score <= -FLT_MAX) {
                exp_score = 0.0f;
            } else {
                exp_score = expf(score - max_val);
            }
            
            float weight = exp_score / sum_exp;
            
            int v_idx = b * head_num * seq_length * head_dim + h * seq_length * head_dim + pos_j * head_dim + d;
            float v_val = __bfloat162float(value[v_idx]);
            sum += weight * v_val;
        }
    }
    
    output[idx] = __float2bfloat16(sum);
}

extern "C" void triangle_attention_bf16(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float* attention_mask,
    __nv_bfloat16* output,
    int batch_size,
    int head_num,
    int seq_length,
    int head_dim
) {
    int total_elements = batch_size * head_num * seq_length * head_dim;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    triangle_attention_bf16_kernel<<<blocks, threads_per_block>>>(
        query, key, value, attention_mask, output,
        batch_size, head_num, seq_length, head_dim
    );
    
    cudaDeviceSynchronize();
}