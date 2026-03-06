#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Tile size for tokens per block
#define TILE_SIZE 8

// Optimized kernel with token-dimension coarsening
__global__ void merge_attn_states_kernel_optimized(
    float* __restrict__ V_out,
    float* __restrict__ LSE_out,
    const float* __restrict__ V_a,
    const float* __restrict__ lse_a,
    const float* __restrict__ V_b,
    const float* __restrict__ lse_b,
    int num_tokens,
    int num_heads,
    int head_size) {
    int vec_count = head_size / 4;
    int tile = blockDim.y;

    // Dynamic shared memory layout: [p_scale[0..tile-1], s_scale[0..tile-1], new_lse[0..tile-1]]
    extern __shared__ float shared_mem[];
    float* p_scale = shared_mem;
    float* s_scale = p_scale + tile;
    float* new_lse = s_scale + tile;

    int block_token_group = blockIdx.x;
    int head_idx = blockIdx.y;
    int lane = threadIdx.x;
    int t = threadIdx.y;
    int token_idx = block_token_group * tile + t;

    // Bounds check
    if (token_idx >= num_tokens || head_idx >= num_heads || lane >= vec_count) {
        return;
    }

    // Compute scales and new LSE per token (one lane per token)
    if (lane == 0) {
        float p_l = __ldg(&lse_a[head_idx * num_tokens + token_idx]);
        float s_l = __ldg(&lse_b[head_idx * num_tokens + token_idx]);
        p_l = isinf(p_l) ? -FLT_MAX : p_l;
        s_l = isinf(s_l) ? -FLT_MAX : s_l;
        float max_l = fmaxf(p_l, s_l);
        float p_e = expf(p_l - max_l);
        float s_e = expf(s_l - max_l);
        float tot = p_e + s_e;
        p_scale[t] = p_e / tot;
        s_scale[t] = s_e / tot;
        new_lse[t] = logf(tot) + max_l;
    }
    __syncthreads();

    // Compute the offset for this float4 vector
    int offset = token_idx * (num_heads * head_size)
               + head_idx * head_size
               + lane * 4;

    // Vectorized loads via read-only cache
    const float4* Va4 = reinterpret_cast<const float4*>(V_a + offset);
    const float4* Vb4 = reinterpret_cast<const float4*>(V_b + offset);
    float4 va = __ldg(Va4);
    float4 vb = __ldg(Vb4);

    // Weighted sum
    float scale_p = p_scale[t];
    float scale_s = s_scale[t];
    float4 vo;
    vo.x = va.x * scale_p + vb.x * scale_s;
    vo.y = va.y * scale_p + vb.y * scale_s;
    vo.z = va.z * scale_p + vb.z * scale_s;
    vo.w = va.w * scale_p + vb.w * scale_s;

    // Store output
    float4* Vo4 = reinterpret_cast<float4*>(V_out + offset);
    *Vo4 = vo;

    // Write back new LSE per token
    if (lane == 0 && LSE_out != nullptr) {
        LSE_out[head_idx * num_tokens + token_idx] = new_lse[t];
    }
}

// External C wrapper with the same signature, launching the optimized kernel
extern "C" void merge_attn_states_optimized(
    float* V_out,
    float* LSE_out,
    const float* V_a,
    const float* lse_a,
    const float* V_b,
    const float* lse_b,
    int num_tokens,
    int num_heads,
    int head_size) {
    int vec_count = head_size / 4;
    int grid_x = (num_tokens + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(grid_x, num_heads, 1);
    dim3 block(vec_count, TILE_SIZE, 1);
    size_t shared_bytes = 3 * TILE_SIZE * sizeof(float);

    merge_attn_states_kernel_optimized<<<grid, block, shared_bytes>>>(
        V_out, LSE_out,
        V_a, lse_a,
        V_b, lse_b,
        num_tokens, num_heads, head_size
    );
    cudaDeviceSynchronize();
}