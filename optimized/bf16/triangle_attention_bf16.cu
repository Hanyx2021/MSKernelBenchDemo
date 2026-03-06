#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

// Fused one-pass streaming softmax + weighted-sum optimized kernel
__global__ void triangle_attention_bf16_kernel_optimized(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float*         attention_mask,
    __nv_bfloat16*       output,
    int                  batch_size,
    int                  head_num,
    int                  seq_length,
    int                  head_dim
) {
    // Block-per-(b,h,i), thread-per-dimension
    int x = blockIdx.x;
    int d = threadIdx.x;

    int bh_seq = head_num * seq_length;
    int b = x / bh_seq;
    int rem = x % bh_seq;
    int h = rem / seq_length;
    int i = rem % seq_length;

    // Compute flattened index for this (b,h,i,d)
    int idx = ((b * head_num + h) * seq_length + i) * head_dim + d;

    // Load and scale query
    float inv_sqrt_dim = 1.0f / sqrtf((float)head_dim);
    float query_val = __bfloat162float(query[idx]);
    float q_scaled = query_val * inv_sqrt_dim;

    // Base offsets
    int base_k = ((b * head_num + h) * seq_length) * head_dim + d;
    int base_v = base_k;
    int base_m = b * seq_length * seq_length + i * seq_length;

    // Streaming softmax accumulators
    float m = -FLT_MAX;
    float s = 0.0f;
    float z = 0.0f;

    // One-pass over j to compute softmax and weighted sum
    for (int j = 0; j <= i; ++j) {
        int keyIdx  = base_k + j * head_dim;
        int valIdx  = base_v + j * head_dim;
        int maskIdx = base_m + j;

        float k_val    = __bfloat162float(key[keyIdx]);
        float mask_val = attention_mask[maskIdx];
        float curr     = (mask_val > 0.5f) ? -FLT_MAX : q_scaled * k_val;
        float v_val    = __bfloat162float(value[valIdx]);

        if (curr > m) {
            // rescale existing accumulators to new max
            float exp_delta = expf(m - curr);
            z = z * exp_delta + v_val;
            s = s * exp_delta + 1.0f;
            m = curr;
        } else {
            float e = expf(curr - m);
            z += e * v_val;
            s += e;
        }
    }

    // Final output
    float result = (s > 0.0f) ? (z / s) : 0.0f;
    output[idx] = __float2bfloat16(result);
}

extern "C" void triangle_attention_bf16_optimized(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float*         attention_mask,
    __nv_bfloat16*       output,
    int                  batch_size,
    int                  head_num,
    int                  seq_length,
    int                  head_dim
) {
    // Launch configuration: one block per (b, h, i), head_dim threads per block
    int grid_x = batch_size * head_num * seq_length;
    dim3 grid(grid_x, 1, 1);
    dim3 block(head_dim, 1, 1);

    triangle_attention_bf16_kernel_optimized<<<grid, block>>>(
        query, key, value, attention_mask, output,
        batch_size, head_num, seq_length, head_dim
    );
    cudaDeviceSynchronize();
}