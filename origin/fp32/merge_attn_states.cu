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

__global__ void merge_attn_states_kernel(
    float* V_out, float* LSE_out, const float* V_a,
    const float* lse_a, const float* V_b,
    const float* lse_b, const int num_tokens, const int num_heads,
    const int head_size) {
    
    const int head_idx = blockIdx.y;
    const int token_idx = blockIdx.x;
    const int element_idx = threadIdx.x;
    
    if (head_idx >= num_heads || token_idx >= num_tokens || element_idx >= head_size) return;
    
    __shared__ float p_scale;
    __shared__ float s_scale;
    __shared__ float new_lse;
    
    if (threadIdx.x == 0) {
        float p_lse = lse_a[head_idx * num_tokens + token_idx];
        float s_lse = lse_b[head_idx * num_tokens + token_idx];
        
        p_lse = isinf(p_lse) ? -FLT_MAX : p_lse;
        s_lse = isinf(s_lse) ? -FLT_MAX : s_lse;
        
        const float max_lse = fmaxf(p_lse, s_lse);
        const float p_exp = expf(p_lse - max_lse);
        const float s_exp = expf(s_lse - max_lse);
        const float total_exp = p_exp + s_exp;
        
        p_scale = p_exp / total_exp;
        s_scale = s_exp / total_exp;
        new_lse = logf(total_exp) + max_lse;
    }
    
    __syncthreads();
    
    const int src_offset = token_idx * num_heads * head_size + 
                          head_idx * head_size + element_idx;
    
    V_out[src_offset] = V_a[src_offset] * p_scale + 
                       V_b[src_offset] * s_scale;
    
    if (LSE_out != nullptr && element_idx == 0) {
        LSE_out[head_idx * num_tokens + token_idx] = new_lse;
    }
}

extern "C" void merge_attn_states(
    float* V_out, 
    float* LSE_out, 
    const float* V_a,
    const float* lse_a, 
    const float* V_b,
    const float* lse_b, 
    int num_tokens, 
    int num_heads,
    int head_size) {
    
    dim3 grid(num_tokens, num_heads, 1);
    dim3 block(head_size, 1, 1);
    
    merge_attn_states_kernel<<<grid, block>>>(V_out, LSE_out, V_a, lse_a, V_b, lse_b, num_tokens, num_heads, head_size);
    cudaDeviceSynchronize();
}