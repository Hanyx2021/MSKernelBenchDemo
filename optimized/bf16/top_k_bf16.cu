#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>      // for INFINITY
#include <climits>
#include <algorithm>

// Structure for value-index pair
struct ValueIndexPairBf16Optimized {
    __nv_bfloat16 value;
    int index;
};

// Convert bf16 to float for comparisons
__device__ inline float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Comparison: return true if a > b, or a == b and a.index < b.index
__device__ inline bool pair_greater(const ValueIndexPairBf16Optimized &a,
                                    const ValueIndexPairBf16Optimized &b) {
    float fa = bf16_to_float(a.value);
    float fb = bf16_to_float(b.value);
    if (fa > fb) return true;
    if (fa < fb) return false;
    // tie-break by smaller index
    return a.index < b.index;
}

// Phase 1: Hierarchical lock-free block-local top-k via warp-level reduction
__global__ void blockLocalTopK_kernel_optimized(
    const __nv_bfloat16* input,
    ValueIndexPairBf16Optimized* block_topk,
    int N, int k) {
    const int localCount = 4;  // each thread loads 4
    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    const int warpSize = 32;
    const int warpId = tid / warpSize;
    const int laneId = tid % warpSize;
    const int W = blockDimX / warpSize;  // warps per block
    int blockStart = blockIdx.x * blockDimX * localCount;

    // Shared buffers
    __shared__ ValueIndexPairBf16Optimized warpSubBuffer[1024];
    __shared__ ValueIndexPairBf16Optimized warpTopKBuffer[1024];

    // Each thread loads localCount elements
    ValueIndexPairBf16Optimized localArr[localCount];
    #pragma unroll
    for (int i = 0; i < localCount; ++i) {
        int idx = blockStart + tid + i * blockDimX;
        if (idx < N) {
            localArr[i].value = input[idx];
            localArr[i].index = idx;
        } else {
            localArr[i].value = __float2bfloat16(-INFINITY);
            localArr[i].index = INT_MAX;
        }
    }
    // sort localArr descending (small fixed size)
    #pragma unroll
    for (int i = 1; i < localCount; ++i) {
        auto key = localArr[i];
        int j = i - 1;
        while (j >= 0 && pair_greater(key, localArr[j])) {
            localArr[j + 1] = localArr[j]; j--;
        }
        localArr[j + 1] = key;
    }
    // write localArr into shared warpSubBuffer
    int warpBase = warpId * warpSize * localCount;
    #pragma unroll
    for (int i = 0; i < localCount; ++i) {
        warpSubBuffer[warpBase + laneId * localCount + i] = localArr[i];
    }
    __syncthreads();

    // Each warp leader builds warp-local top-k by scanning full segment
    int warpSegSize = warpSize * localCount;
    if (laneId == 0) {
        int topCount = k;
        ValueIndexPairBf16Optimized tmp[128];  // k <= 128
        int curCount = 0;
        for (int i = 0; i < warpSegSize; ++i) {
            auto v = warpSubBuffer[warpBase + i];
            if (curCount < topCount) {
                tmp[curCount++] = v;
                if (curCount == topCount) {
                    // initial insertion sort
                    for (int p = 1; p < topCount; ++p) {
                        auto key = tmp[p];
                        int q = p - 1;
                        while (q >= 0 && pair_greater(key, tmp[q])) {
                            tmp[q+1] = tmp[q]; q--;
                        }
                        tmp[q+1] = key;
                    }
                }
            } else if (pair_greater(v, tmp[topCount - 1])) {
                // replace smallest and re-insert
                tmp[topCount - 1] = v;
                int q = topCount - 2;
                while (q >= 0 && pair_greater(tmp[q+1], tmp[q])) {
                    auto sw = tmp[q]; tmp[q] = tmp[q+1]; tmp[q+1] = sw;
                    q--;
                }
            }
        }
        // write out this warp's top-k into warpTopKBuffer
        int warpKbase = warpId * k;
        for (int i = 0; i < topCount; ++i) {
            warpTopKBuffer[warpKbase + i] = tmp[i];
        }
    }
    __syncthreads();

    // Block-level merge: gather all W*k candidates
    __shared__ ValueIndexPairBf16Optimized mergeBuf[1024];
    int idxBuf = tid;
    int totalWarpCand = W * k;
    while (idxBuf < totalWarpCand) {
        mergeBuf[idxBuf] = warpTopKBuffer[idxBuf];
        idxBuf += blockDimX;
    }
    __syncthreads();

    // Simple single-threaded selection: pick top-k out of total candidates
    if (tid == 0) {
        for (int i = 0; i < k; ++i) {
            int best = i;
            for (int j = i + 1; j < totalWarpCand; ++j) {
                if (pair_greater(mergeBuf[j], mergeBuf[best])) {
                    best = j;
                }
            }
            auto tmp = mergeBuf[i];
            mergeBuf[i] = mergeBuf[best];
            mergeBuf[best] = tmp;
        }
    }
    __syncthreads();

    // write out the top-k into block_topk
    int outBase = blockIdx.x * k;
    int lane = tid % warpSize;
    for (int j = lane; j < k; j += warpSize) {
        block_topk[outBase + j] = mergeBuf[j];
    }
}

// Phase 2: Parallel Multi-Threaded Global Merge via Tree-Structured k-Way Reduction
__global__ void top_k_bf16_optimized(
    const ValueIndexPairBf16Optimized* block_topk,
    __nv_bfloat16* top_k_values,
    int* top_k_indices,
    int B, int k) {
    int lane = threadIdx.x;
    if (lane >= k) return;
    extern __shared__ ValueIndexPairBf16Optimized sh_mem[];
    // sh_mem[0..k-1] for S, sh_mem[k..2k-1] for buffer/T
    ValueIndexPairBf16Optimized* S   = sh_mem;
    ValueIndexPairBf16Optimized* buf = sh_mem + k;

    // Phase 0: load first block's k results into S
    S[lane] = block_topk[lane];
    __syncthreads();

    // Merge each subsequent block's top-k
    for (int i = 1; i < B; ++i) {
        // load block i's top-k list into buf
        buf[lane] = block_topk[i * k + lane];
        __syncthreads();
        
        // parallel merge of S and buf into buf
        // Each lane handles one element from S and one from buf
        ValueIndexPairBf16Optimized a = S[lane];
        // binary search in buf for number of elements > a
        int low = 0, high = k;
        while (low < high) {
            int mid = (low + high) >> 1;
            if (pair_greater(buf[mid], a)) low = mid + 1;
            else high = mid;
        }
        int posA = lane + low;

        ValueIndexPairBf16Optimized b = buf[lane];
        // binary search in S for number of elements > b
        low = 0; high = k;
        while (low < high) {
            int mid = (low + high) >> 1;
            if (pair_greater(S[mid], b)) low = mid + 1;
            else high = mid;
        }
        int posB = lane + low;

        // write merged results to buf as temporary
        if (posA < k) buf[posA] = a;
        if (posB < k) buf[posB] = b;
        __syncthreads();

        // copy buf back to S for next iteration
        S[lane] = buf[lane];
        __syncthreads();
    }

    // write final top-k results
    top_k_values[lane]  = S[lane].value;
    top_k_indices[lane] = S[lane].index;
}

extern "C" void top_k_bf16_optimized(
    __nv_bfloat16* top_k_values,
    int* top_k_indices,
    const __nv_bfloat16* input,
    const int N,
    const int k) {
    const int blockDimX = 256;
    const int localCount = 4;
    int chunkSize = blockDimX * localCount;
    int B = (N + chunkSize - 1) / chunkSize;

    // Allocate block-level top-k buffer
    ValueIndexPairBf16Optimized* d_block_topk;
    cudaMalloc(&d_block_topk, B * k * sizeof(ValueIndexPairBf16Optimized));

    // Launch block-local top-k kernel
    dim3 block(blockDimX);
    dim3 grid(B);
    blockLocalTopK_kernel_optimized<<<grid, block>>>(input, d_block_topk, N, k);
    cudaDeviceSynchronize();

    // Phase 2: parallel global merge
    int threads = 1;
    while (threads < k) threads <<= 1;
    if (threads > 1024) threads = 1024;
    size_t shared_mem = 2 * k * sizeof(ValueIndexPairBf16Optimized);
    top_k_bf16_optimized<<<1, threads, shared_mem>>>(
        d_block_topk, top_k_values, top_k_indices, B, k);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_block_topk);
}