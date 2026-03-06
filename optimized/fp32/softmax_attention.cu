#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Q*K^T kernel (unchanged)
__global__ void qkT_kernel_optimized(int q_seq_len, int kv_seq_len, int dim_qk,
                                     const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     float* __restrict__ S) {
    const int TH = 16;
    const int TW = 32;
    const int TK = 32;

    int row = blockIdx.y * TH + threadIdx.y;
    int col = blockIdx.x * TW + threadIdx.x;

    float acc = 0.0f;

    __shared__ float shareQ[TH][TK];
    __shared__ float shareK[TK][TW];

    int numTiles = (dim_qk + TK - 1) / TK;
    for (int t = 0; t < numTiles; ++t) {
        int kStart = t * TK;
        if (threadIdx.x < (TW / 2)) {
            int c0 = kStart + threadIdx.x * 2;
            if (row < q_seq_len) {
                if (c0 + 1 < dim_qk) {
                    float2 v = *(reinterpret_cast<const float2*>(Q + row * dim_qk + c0));
                    shareQ[threadIdx.y][threadIdx.x * 2]     = v.x;
                    shareQ[threadIdx.y][threadIdx.x * 2 + 1] = v.y;
                } else if (c0 < dim_qk) {
                    shareQ[threadIdx.y][threadIdx.x * 2]     = Q[row * dim_qk + c0];
                    shareQ[threadIdx.y][threadIdx.x * 2 + 1] = 0.0f;
                } else {
                    shareQ[threadIdx.y][threadIdx.x * 2]     = 0.0f;
                    shareQ[threadIdx.y][threadIdx.x * 2 + 1] = 0.0f;
                }
            } else {
                shareQ[threadIdx.y][threadIdx.x * 2]     = 0.0f;
                shareQ[threadIdx.y][threadIdx.x * 2 + 1] = 0.0f;
            }
        }
        if (threadIdx.y < (TK / 2)) {
            int r0 = kStart + threadIdx.y * 2;
            if (col < kv_seq_len) {
                if (r0 + 1 < dim_qk) {
                    float2 v = *(reinterpret_cast<const float2*>(K + col * dim_qk + r0));
                    shareK[threadIdx.y * 2][threadIdx.x]     = v.x;
                    shareK[threadIdx.y * 2 + 1][threadIdx.x] = v.y;
                } else if (r0 < dim_qk) {
                    shareK[threadIdx.y * 2][threadIdx.x]     = K[col * dim_qk + r0];
                    shareK[threadIdx.y * 2 + 1][threadIdx.x] = 0.0f;
                } else {
                    shareK[threadIdx.y * 2][threadIdx.x]     = 0.0f;
                    shareK[threadIdx.y * 2 + 1][threadIdx.x] = 0.0f;
                }
            } else {
                shareK[threadIdx.y * 2][threadIdx.x]     = 0.0f;
                shareK[threadIdx.y * 2 + 1][threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();
        #pragma unroll 4
        for (int k = 0; k < TK; ++k) {
            acc += shareQ[threadIdx.y][k] * shareK[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < q_seq_len && col < kv_seq_len) {
        float scale = rsqrtf((float)dim_qk);
        S[row * kv_seq_len + col] = acc * scale;
    }
}

// Warp-level reduction helpers (warp size is a known 32 lanes)
static __inline__ __device__ float warp_reduce_max(float val) {
    // 32-lane warp => offsets 16,8,4,2,1
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
static __inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized block-wise softmax kernel
__global__ void softmax_kernel_optimized(
    float* out,
    const float* __restrict__ input,
    int q_seq_len,
    int kv_seq_len) {
    int row = blockIdx.x;
    if (row >= q_seq_len) return;
    extern __shared__ float sdata[];

    const float* input_row = input + row * kv_seq_len;
    float* out_row = out + row * kv_seq_len;

    int tid  = threadIdx.x;
    // warpSize==32 => low 5 bits are lane id, high bits are warp id
    int lane = tid & (32 - 1);
    int wid  = tid >> 5; // warp id

    // Phase 1: compute max
    float thread_max = -FLT_MAX;
    int elems_per_thread = 4;
    int idx = tid * elems_per_thread;
    int stride = blockDim.x * elems_per_thread;
    int num_vec = kv_seq_len / elems_per_thread;
    // vectorized loads
    for (int i = idx; i < num_vec * elems_per_thread; i += stride) {
        float4 v = reinterpret_cast<const float4*>(input_row)[i / elems_per_thread];
        thread_max = fmaxf(thread_max, v.x);
        thread_max = fmaxf(thread_max, v.y);
        thread_max = fmaxf(thread_max, v.z);
        thread_max = fmaxf(thread_max, v.w);
    }
    // tail
    for (int i = num_vec * elems_per_thread + tid; i < kv_seq_len; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input_row[i]);
    }
    // warp reduction
    float warp_max = warp_reduce_max(thread_max);
    // inter-warp
    if (lane == 0) sdata[wid] = warp_max;
    __syncthreads();
    float block_max = -FLT_MAX;
    if (wid == 0) {
        float v = (tid < (blockDim.x >> 5)) ? sdata[tid] : -FLT_MAX;
        float warp_leader_max = warp_reduce_max(v);
        if (tid == 0) sdata[0] = warp_leader_max;
    }
    __syncthreads();
    block_max = sdata[0];

    // Phase 2: compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int i = idx; i < num_vec * elems_per_thread; i += stride) {
        float4 v = reinterpret_cast<const float4*>(input_row)[i / elems_per_thread];
        thread_sum += expf(v.x - block_max);
        thread_sum += expf(v.y - block_max);
        thread_sum += expf(v.z - block_max);
        thread_sum += expf(v.w - block_max);
    }
    for (int i = num_vec * elems_per_thread + tid; i < kv_seq_len; i += blockDim.x) {
        thread_sum += expf(input_row[i] - block_max);
    }
    // warp reduction
    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) sdata[wid] = warp_sum;
    __syncthreads();
    float block_sum = 0.0f;
    if (wid == 0) {
        float v = (tid < (blockDim.x >> 5)) ? sdata[tid] : 0.0f;
        float warp_leader_sum = warp_reduce_sum(v);
        if (tid == 0) sdata[0] = warp_leader_sum;
    }
    __syncthreads();
    block_sum = sdata[0];

    // Phase 3: write output
    for (int i = idx; i < num_vec * elems_per_thread; i += stride) {
        float4 v = reinterpret_cast<const float4*>(input_row)[i / elems_per_thread];
        float e0 = expf(v.x - block_max) / block_sum;
        float e1 = expf(v.y - block_max) / block_sum;
        float e2 = expf(v.z - block_max) / block_sum;
        float e3 = expf(v.w - block_max) / block_sum;
        reinterpret_cast<float4*>(out_row)[i / elems_per_thread] = make_float4(e0, e1, e2, e3);
    }
    for (int i = num_vec * elems_per_thread + tid; i < kv_seq_len; i += blockDim.x) {
        out_row[i] = expf(input_row[i] - block_max) / block_sum;
    }
}

// S*V kernel (unchanged)
__global__ void sv_kernel_optimized(int q_seq_len, int kv_seq_len, int dim_v,
                                    const float* __restrict__ S,
                                    const float* __restrict__ V,
                                    float* __restrict__ Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= q_seq_len || col >= dim_v) return;

    float acc = 0.0f;
    for (int k = 0; k < kv_seq_len; ++k) {
        acc += S[row * kv_seq_len + k] * V[k * dim_v + col];
    }
    Y[row * dim_v + col] = acc;
}

// External C wrapper
extern "C" void softmax_attention_optimized(
    float* Y,
    const float* Q,
    const float* K,
    const float* V,
    int q_seq_len,
    int kv_seq_len,
    int dim_qk,
    int dim_v) {

    float* S = nullptr;
    float* S_softmax = nullptr;
    size_t S_size = static_cast<size_t>(q_seq_len) * kv_seq_len * sizeof(float);
    cudaMalloc(&S, S_size);
    cudaMalloc(&S_softmax, S_size);

    // Launch Q*K^T kernel
    const int TH = 16;
    const int TW = 32;
    dim3 block_qk(TW, TH);
    dim3 grid_qk((kv_seq_len + TW - 1) / TW,
                 (q_seq_len + TH - 1) / TH);
    qkT_kernel_optimized<<<grid_qk, block_qk>>>(
        q_seq_len, kv_seq_len, dim_qk, Q, K, S);

    // Launch optimized softmax kernel: one block per row
    const int SOFT_MAX_THD = 128;
    const int WARP_SIZE = 32;
    dim3 grid_sm(q_seq_len);
    softmax_kernel_optimized<<<grid_sm, SOFT_MAX_THD,
        (SOFT_MAX_THD / WARP_SIZE) * sizeof(float)>>>(
        S_softmax, S, q_seq_len, kv_seq_len);

    // Launch S*V kernel
    dim3 block_sv(16, 16);
    dim3 grid_sv((dim_v + block_sv.x - 1) / block_sv.x,
                 (q_seq_len + block_sv.y - 1) / block_sv.y);
    sv_kernel_optimized<<<grid_sv, block_sv>>>(
        q_seq_len, kv_seq_len, dim_v, S_softmax, V, Y);

    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(S_softmax);
}