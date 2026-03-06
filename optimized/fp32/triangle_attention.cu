#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

// Optimized triangle attention kernel
__global__ void triangle_attention_kernel_optimized(
    const float* query,
    const float* key,
    const float* value,
    const float* attention_mask,
    float* output,
    int batch_size,
    int head_num,
    int seq_length,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * head_num * seq_length * head_dim;
    if (idx >= total_elements) return;

    // Decode indices
    int b = idx / (head_num * seq_length * head_dim);
    int rem = idx % (head_num * seq_length * head_dim);
    int h = rem / (seq_length * head_dim);
    rem = rem % (seq_length * head_dim);
    int pos_i = rem / head_dim;
    int d = rem % head_dim;

    // Hoist invariant computations
    float inv_scale = rsqrtf((float)head_dim);
    float q_val = query[idx];

    // First pass: compute max score
    float max_val = -INFINITY;
    for (int pos_j = 0; pos_j <= pos_i; ++pos_j) {
        int k_idx = b * head_num * seq_length * head_dim
                    + h * seq_length * head_dim
                    + pos_j * head_dim + d;
        float k_val = key[k_idx];
        float score = q_val * k_val * inv_scale;
        int mask_idx = b * seq_length * seq_length
                       + pos_i * seq_length + pos_j;
        if (attention_mask[mask_idx] > 0.5f) {
            score = -INFINITY;
        }
        max_val = fmaxf(score, max_val);
    }

    // Second pass: compute sum of exponentials
    float sum_exp = 0.0f;
    for (int pos_j = 0; pos_j <= pos_i; ++pos_j) {
        int k_idx = b * head_num * seq_length * head_dim
                    + h * seq_length * head_dim
                    + pos_j * head_dim + d;
        float k_val = key[k_idx];
        float score = q_val * k_val * inv_scale;
        int mask_idx = b * seq_length * seq_length
                       + pos_i * seq_length + pos_j;
        if (attention_mask[mask_idx] > 0.5f) {
            score = -INFINITY;
        }
        float exp_score = (score <= -INFINITY)
                          ? 0.0f
                          : expf(score - max_val);
        sum_exp += exp_score;
    }

    // Third pass: weighted sum of values
    float sum = 0.0f;
    if (sum_exp > 0.0f) {
        for (int pos_j = 0; pos_j <= pos_i; ++pos_j) {
            int k_idx = b * head_num * seq_length * head_dim
                        + h * seq_length * head_dim
                        + pos_j * head_dim + d;
            float k_val = key[k_idx];
            float score = q_val * k_val * inv_scale;
            int mask_idx = b * seq_length * seq_length
                           + pos_i * seq_length + pos_j;
            if (attention_mask[mask_idx] > 0.5f) {
                score = -INFINITY;
            }
            float exp_score = (score <= -INFINITY)
                              ? 0.0f
                              : expf(score - max_val);
            float weight = exp_score / sum_exp;
            int v_idx = b * head_num * seq_length * head_dim
                        + h * seq_length * head_dim
                        + pos_j * head_dim + d;
            sum += weight * value[v_idx];
        }
    }
    output[idx] = sum;
}

extern "C" void triangle_attention_optimized(
    const float* query,
    const float* key,
    const float* value,
    const float* attention_mask,
    float* output,
    int batch_size,
    int head_num,
    int seq_length,
    int head_dim
) {
    int total_elements = batch_size * head_num * seq_length * head_dim;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    triangle_attention_kernel_optimized<<<blocks, threads_per_block>>>(
        query, key, value, attention_mask, output,
        batch_size, head_num, seq_length, head_dim
    );
    cudaDeviceSynchronize();
}