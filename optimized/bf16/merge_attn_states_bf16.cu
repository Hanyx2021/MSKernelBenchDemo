#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

// Optimized kernel using read-only data cache and __restrict__ pointers
__global__ void merge_attn_states_bf16_kernel_optimized(
    __nv_bfloat162* __restrict__ V_out,
    __nv_bfloat16* __restrict__ LSE_out,
    const __nv_bfloat162* __restrict__ V_a,
    const __nv_bfloat16* __restrict__ lse_a,
    const __nv_bfloat162* __restrict__ V_b,
    const __nv_bfloat16* __restrict__ lse_b,
    int num_tokens,
    int num_heads,
    int head_size) {
    int head_idx = blockIdx.y;
    int token_idx = blockIdx.x;
    int pair_idx = threadIdx.x;

    int head_pairs = (head_size + 1) / 2;
    if (head_idx >= num_heads || token_idx >= num_tokens || pair_idx >= head_pairs) return;

    __shared__ float p_scale, s_scale, new_lse;
    int lse_index = head_idx * num_tokens + token_idx;
    if (pair_idx == 0) {
        // Load lse values through read-only cache
        float p_lse = __bfloat162float(__ldg(&lse_a[lse_index]));
        float s_lse = __bfloat162float(__ldg(&lse_b[lse_index]));
        // handle infinities
        if (isinf(p_lse)) p_lse = -FLT_MAX;
        if (isinf(s_lse)) s_lse = -FLT_MAX;
        float max_lse = fmaxf(p_lse, s_lse);
        float p_exp = expf(p_lse - max_lse);
        float s_exp = expf(s_lse - max_lse);
        float total = p_exp + s_exp;
        p_scale = p_exp / total;
        s_scale = s_exp / total;
        new_lse = logf(total) + max_lse;
    }
    __syncthreads();

    // Compute offset in vectorized array
    int pair_offset = token_idx * (num_heads * head_pairs)
                      + head_idx * head_pairs + pair_idx;

    // Load bf16 pairs from inputs via read-only cache
    __nv_bfloat162 va_pair = __ldg(&V_a[pair_offset]);
    __nv_bfloat162 vb_pair = __ldg(&V_b[pair_offset]);

    // Unpack to floats and compute weighted sum
    float va0 = __bfloat162float(va_pair.x);
    float vb0 = __bfloat162float(vb_pair.x);
    float r0 = va0 * p_scale + vb0 * s_scale;
    __nv_bfloat16 out0 = __float2bfloat16(r0);

    // Check if second element exists (odd head_size guard)
    bool has_second = (pair_idx * 2 + 1 < head_size);
    __nv_bfloat16 out1;
    if (has_second) {
        float va1 = __bfloat162float(va_pair.y);
        float vb1 = __bfloat162float(vb_pair.y);
        float r1 = va1 * p_scale + vb1 * s_scale;
        out1 = __float2bfloat16(r1);
    } else {
        out1 = va_pair.y;  // padding for odd element
    }

    // Pack and store result
    __nv_bfloat162 out_pair;
    out_pair.x = out0;
    out_pair.y = out1;
    V_out[pair_offset] = out_pair;

    // Write new LSE once per token-head
    if (LSE_out != nullptr && pair_idx == 0) {
        LSE_out[lse_index] = __float2bfloat16(new_lse);
    }
}

// External C wrapper with optimized kernel invocation
extern "C" void merge_attn_states_bf16_optimized(
    __nv_bfloat16* __restrict__ V_out,
    __nv_bfloat16* __restrict__ LSE_out,
    const __nv_bfloat16* __restrict__ V_a,
    const __nv_bfloat16* __restrict__ lse_a,
    const __nv_bfloat16* __restrict__ V_b,
    const __nv_bfloat16* __restrict__ lse_b,
    int num_tokens,
    int num_heads,
    int head_size) {
    int head_pairs = (head_size + 1) / 2;
    dim3 grid(num_tokens, num_heads);
    dim3 block(head_pairs);

    // Reinterpret raw bf16 pointers as bf16-pair pointers
    auto V_a_pair = reinterpret_cast<const __nv_bfloat162* __restrict__>(V_a);
    auto V_b_pair = reinterpret_cast<const __nv_bfloat162* __restrict__>(V_b);
    auto V_out_pair = reinterpret_cast<__nv_bfloat162* __restrict__>(V_out);

    merge_attn_states_bf16_kernel_optimized<<<grid, block>>>(
        V_out_pair, LSE_out,
        V_a_pair, lse_a,
        V_b_pair, lse_b,
        num_tokens, num_heads, head_size);
    cudaDeviceSynchronize();
}