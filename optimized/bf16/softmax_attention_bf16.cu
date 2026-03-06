#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <float.h>

// Configuration struct for optimized softmax attention
struct SoftmaxAttentionBf16OptimizedConfig {
    dim3 block_qk;
    dim3 grid_qk;
    dim3 block_sv;
    dim3 grid_sv;
    int threadsPerBlockSoftmax;
    int blocksSoftmax;

    SoftmaxAttentionBf16OptimizedConfig(int q_seq_len, int kv_seq_len, int dim_v) {
        // use a 16×16 thread-tile
        dim3 block2d(16, 16, 1);

        block_qk = block2d;
        grid_qk  = dim3(
            (kv_seq_len + block2d.x - 1) / block2d.x,
            (q_seq_len  + block2d.y - 1) / block2d.y,
            1
        );

        block_sv = block2d;
        grid_sv  = dim3(
            (dim_v     + block2d.x - 1) / block2d.x,
            (q_seq_len + block2d.y - 1) / block2d.y,
            1
        );

        threadsPerBlockSoftmax = 256;
        blocksSoftmax = (q_seq_len + threadsPerBlockSoftmax - 1) / threadsPerBlockSoftmax;
    }
};

// QK^T kernel using simple loops (fallback implementation)
__global__ void qkT_bf16_kernel_optimized(
    int q_seq_len,
    int kv_seq_len,
    int dim_qk,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    __nv_bfloat16* __restrict__ S)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= q_seq_len || col >= kv_seq_len) return;

    float acc = 0.0f;
    // dot product Q[row,:] · K[col,:]
    for (int k = 0; k < dim_qk; ++k) {
        acc += __bfloat162float(Q[row * dim_qk + k])
             * __bfloat162float(K[col * dim_qk + k]);
    }
    float scale = rsqrtf((float)dim_qk);
    S[row * kv_seq_len + col] = __float2bfloat16(acc * scale);
}

// Optimized Softmax kernel with warp-shuffle and vectorized loads/stores
#define WARP_SIZE 32
#define CHUNK_SIZE 4

__global__ void softmax_bf16_kernel_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int /*q_seq_len*/,
    int kv_seq_len)
{
    int row = blockIdx.x;
    int lane = threadIdx.x;
    const __nv_bfloat16* input_row = input + row * kv_seq_len;
    __nv_bfloat16* out_row = out + row * kv_seq_len;

    // Phase 1: find max
    float local_max = -FLT_MAX;
    int stride = WARP_SIZE * CHUNK_SIZE;
    for (int base = lane * CHUNK_SIZE; base < kv_seq_len; base += stride) {
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            int idx = base + i;
            float v = -FLT_MAX;
            if (idx < kv_seq_len) {
                __nv_bfloat16 tmp = __ldg(&input_row[idx]);
                v = __bfloat162float(tmp);
            }
            local_max = fmaxf(local_max, v);
        }
    }
    // Warp-wide reduction for max
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    float max_val = __shfl_sync(0xffffffff, local_max, 0);

    // Phase 2: compute exp and sum
    float local_sum = 0.0f;
    // We store exp values in registers per chunk
    for (int base = lane * CHUNK_SIZE; base < kv_seq_len; base += stride) {
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            int idx = base + i;
            if (idx < kv_seq_len) {
                float v = __bfloat162float(__ldg(&input_row[idx]));
                float e = expf(v - max_val);
                local_sum += e;
                // temporarily write the scaled exp back to out (reuse out buffer as staging)
                out_row[idx] = __float2bfloat16(e);
            }
        }
    }
    // Warp-wide reduction for sum
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    float inv_sum = 1.0f / __shfl_sync(0xffffffff, local_sum, 0);

    // Phase 3: write final softmax outputs
    for (int base = lane * CHUNK_SIZE; base < kv_seq_len; base += stride) {
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            int idx = base + i;
            if (idx < kv_seq_len) {
                float e = __bfloat162float(out_row[idx]);
                out_row[idx] = __float2bfloat16(e * inv_sum);
            }
        }
    }
}

// SV kernel using simple loops (fallback implementation)
__global__ void sv_bf16_kernel_optimized(
    int q_seq_len,
    int kv_seq_len,
    int dim_v,
    const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ Y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= q_seq_len || col >= dim_v) return;

    float acc = 0.0f;
    for (int k = 0; k < kv_seq_len; ++k) {
        acc += __bfloat162float(S[row * kv_seq_len + k])
             * __bfloat162float(V[k * dim_v + col]);
    }
    Y[row * dim_v + col] = __float2bfloat16(acc);
}

// External C wrapper
extern "C" void softmax_attention_bf16_optimized(
    __nv_bfloat16* Y,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    int q_seq_len,
    int kv_seq_len,
    int dim_qk,
    int dim_v)
{
    __nv_bfloat16* S = nullptr;
    __nv_bfloat16* S_softmax = nullptr;
    size_t S_size = size_t(q_seq_len) * size_t(kv_seq_len) * sizeof(__nv_bfloat16);
    cudaMalloc(&S, S_size);
    cudaMalloc(&S_softmax, S_size);

    SoftmaxAttentionBf16OptimizedConfig cfg(q_seq_len, kv_seq_len, dim_v);

    qkT_bf16_kernel_optimized<<<cfg.grid_qk, cfg.block_qk>>>(
        q_seq_len, kv_seq_len, dim_qk, Q, K, S);

    // Launch optimized softmax: one block per row, one warp per block
    int grid = q_seq_len;
    int block = WARP_SIZE;
    softmax_bf16_kernel_optimized<<<grid, block>>>(
        S_softmax, S, q_seq_len, kv_seq_len);

    sv_bf16_kernel_optimized<<<cfg.grid_sv, cfg.block_sv>>>(
        q_seq_len, kv_seq_len, dim_v, S_softmax, V, Y);

    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(S_softmax);
}
