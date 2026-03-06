#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>

// Value-index pair
struct ValueIndexPairOptimized {
    float value;
    int index;
};

// Phase-1 Kernel: Each block sorts its chunk of size BLOCK_SIZE and writes top-k
template <int BLOCK_SIZE>
__global__ void block_local_topk_kernel_optimized(
    const float* __restrict__ input,
    ValueIndexPairOptimized* __restrict__ block_results,
    int N,
    int k)
{
    __shared__ ValueIndexPairOptimized s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    ValueIndexPairOptimized vip;
    if (gid < N) {
        vip.value = input[gid];
        vip.index = gid;
    } else {
        vip.value = -FLT_MAX;
        vip.index = INT_MAX;
    }
    s_data[tid] = vip;
    __syncthreads();

    // Bitonic sort in shared memory (descending order)
    for (int size = 2; size <= BLOCK_SIZE; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int ixj = tid ^ stride;
            if (ixj > tid) {
                bool up = ((tid & size) == 0);
                ValueIndexPairOptimized a = s_data[tid];
                ValueIndexPairOptimized b = s_data[ixj];
                bool do_swap = false;
                if (up) {
                    if (a.value < b.value || (a.value == b.value && a.index > b.index)) do_swap = true;
                } else {
                    if (a.value > b.value || (a.value == b.value && a.index < b.index)) do_swap = true;
                }
                if (do_swap) {
                    s_data[tid] = b;
                    s_data[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // Write top-k from each block
    if (tid < k) {
        block_results[blockIdx.x * k + tid] = s_data[tid];
    }
}

// Phase-2 Kernel: Merge G sorted lists of size k into top-k
__global__ void merge_topk_kernel_optimized(
    const ValueIndexPairOptimized* __restrict__ input_lists,
    ValueIndexPairOptimized* __restrict__ output_lists,
    int num_lists,
    int k,
    int G)
{
    extern __shared__ ValueIndexPairOptimized s_data[];
    int tid = threadIdx.x;
    int group_id = blockIdx.x;
    int first = group_id * G;
    int curG = (first + G <= num_lists) ? G : (num_lists - first);
    int total = curG * k;
    int p2 = blockDim.x;  // next power-of-two >= G*k

    // Load and pad
    for (int i = tid; i < p2; i += p2) {
        if (i < total) {
            s_data[i] = input_lists[first * k + i];
        } else {
            s_data[i].value = -FLT_MAX;
            s_data[i].index = INT_MAX;
        }
    }
    __syncthreads();

    // Bitonic sort across p2 elements (descending)
    for (int size = 2; size <= p2; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int ixj = tid ^ stride;
            if (ixj > tid) {
                bool up = ((tid & size) == 0);
                auto a = s_data[tid];
                auto b = s_data[ixj];
                bool do_swap = false;
                if (up) {
                    if (a.value < b.value || (a.value == b.value && a.index > b.index)) do_swap = true;
                } else {
                    if (a.value > b.value || (a.value == b.value && a.index < b.index)) do_swap = true;
                }
                if (do_swap) {
                    s_data[tid] = b;
                    s_data[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // Write top-k from merged group
    for (int i = tid; i < k; i += p2) {
        output_lists[group_id * k + i] = s_data[i];
    }
}

// Kernel to write final top-k to output arrays
__global__ void write_topk_kernel_optimized(
    const ValueIndexPairOptimized* __restrict__ final_list,
    float* top_k_values,
    int* top_k_indices,
    int k)
{
    int i = threadIdx.x;
    if (i < k) {
        top_k_values[i] = final_list[i].value;
        top_k_indices[i] = final_list[i].index;
    }
}

// External C wrapper
extern "C" void top_k_optimized(
    float* top_k_values,
    int* top_k_indices,
    const float* input,
    const int N,
    const int k)
{
    const int BLOCK_SIZE = 256;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (numBlocks <= 0 || k <= 0) return;

    // Allocate buffer for block-local top-k
    ValueIndexPairOptimized* d_block_results = nullptr;
    cudaMalloc(&d_block_results, numBlocks * k * sizeof(ValueIndexPairOptimized));

    // Phase 1: per-block local top-k
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(numBlocks);
    block_local_topk_kernel_optimized<BLOCK_SIZE><<<gridDim, blockDim>>>(
        input, d_block_results, N, k);
    cudaDeviceSynchronize();

    // Allocate temporary buffer for merging stages
    ValueIndexPairOptimized* d_temp = nullptr;
    cudaMalloc(&d_temp, numBlocks * k * sizeof(ValueIndexPairOptimized));

    // Merge tree parameters
    int current_lists = numBlocks;
    ValueIndexPairOptimized* d_in = d_block_results;
    ValueIndexPairOptimized* d_out = d_temp;

    // Helper for next power of two
    auto nextPow2 = [](int v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    };

    // Multi-stage merge until single list remains
    while (current_lists > 1) {
        // Determine group size G
        int maxG1 = 49152 / (k * sizeof(ValueIndexPairOptimized));
        int maxG2 = 1024 / k;
        int G = maxG1 < maxG2 ? maxG1 : maxG2;
        G = (G < 16) ? G : 16;
        if (G < 2) G = 2;

        int groups = (current_lists + G - 1) / G;
        int p2 = nextPow2(G * k);
        size_t smem_size = (size_t)G * k * sizeof(ValueIndexPairOptimized);

        // Launch merge kernel
        merge_topk_kernel_optimized<<<groups, p2, smem_size>>>(
            d_in, d_out, current_lists, k, G);
        cudaDeviceSynchronize();

        // Swap buffers
        current_lists = groups;
        ValueIndexPairOptimized* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    // Write final top-k to output
    int threads_final = nextPow2(k);
    write_topk_kernel_optimized<<<1, threads_final>>>(
        d_in, top_k_values, top_k_indices, k);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_block_results);
    cudaFree(d_temp);
}
