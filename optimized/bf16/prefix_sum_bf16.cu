#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>

#define BLOCK_SIZE 512

// Phase 1: Intra-block scan (Blelloch) on input bf16 -> temp float, and compute block sums
__global__ void prefix_sum_bf16_phase1_kernel_optimized(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ temp,
    float* __restrict__ block_sums,
    int N) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * BLOCK_SIZE + tid;

    // load to shared memory
    float val = 0.0f;
    if (gid < N) val = __bfloat162float(input[gid]);
    sdata[tid] = val;
    __syncthreads();

    // up-sweep (reduce) phase
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < BLOCK_SIZE) {
            sdata[idx] += sdata[idx - offset];
        }
        __syncthreads();
    }

    // save total sum for this block and set root to zero for exclusive scan
    if (tid == 0) {
        block_sums[bid] = sdata[BLOCK_SIZE - 1];
        sdata[BLOCK_SIZE - 1] = 0.0f;
    }
    __syncthreads();

    // down-sweep phase
    for (int offset = BLOCK_SIZE >> 1; offset >= 1; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < BLOCK_SIZE) {
            float t = sdata[idx - offset];
            sdata[idx - offset] = sdata[idx];
            sdata[idx] += t;
        }
        __syncthreads();
    }

    // write intra-block scan result to temp (turn exclusive to inclusive)
    if (gid < N) {
        temp[gid] = sdata[tid] + val;
    }
}

// Phase 2: Exclusive scan of block sums array
__global__ void prefix_sum_bf16_phase2_kernel_optimized(
    const float* __restrict__ block_sums,
    float* __restrict__ block_scanned,
    int numBlocks) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    // load and pad
    if (tid < numBlocks) sdata[tid] = block_sums[tid];
    else                sdata[tid] = 0.0f;
    __syncthreads();

    // up-sweep phase
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < BLOCK_SIZE) {
            sdata[idx] += sdata[idx - offset];
        }
        __syncthreads();
    }

    // set root to zero for exclusive scan
    if (tid == 0) {
        sdata[BLOCK_SIZE - 1] = 0.0f;
    }
    __syncthreads();

    // down-sweep phase
    for (int offset = BLOCK_SIZE >> 1; offset >= 1; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < BLOCK_SIZE) {
            float t = sdata[idx - offset];
            sdata[idx - offset] = sdata[idx];
            sdata[idx] += t;
        }
        __syncthreads();
    }

    // write results back
    if (tid < numBlocks) {
        block_scanned[tid] = sdata[tid];
    }
}

// Phase 3: Add block-wise offsets and convert back to bf16
__global__ void prefix_sum_bf16_phase3_kernel_optimized(
    const float* __restrict__ temp,
    const float* __restrict__ block_scanned,
    __nv_bfloat16* __restrict__ output,
    int N) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * BLOCK_SIZE + tid;
    if (gid < N) {
        float val = temp[gid] + block_scanned[bid];
        output[gid] = __float2bfloat16(val);
    }
}

// External C wrapper
extern "C" void prefix_sum_bf16_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int N) {
    const int threads_per_block = BLOCK_SIZE;
    const int blocks_per_grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate intermediates
    float* d_temp = nullptr;
    float* d_block_sums = nullptr;
    float* d_block_scanned = nullptr;
    cudaMalloc(&d_temp, N * sizeof(float));
    cudaMalloc(&d_block_sums, blocks_per_grid * sizeof(float));
    cudaMalloc(&d_block_scanned, blocks_per_grid * sizeof(float));

    // Phase 1
    prefix_sum_bf16_phase1_kernel_optimized<<<blocks_per_grid, threads_per_block>>>(
        input, d_temp, d_block_sums, N);
    // Phase 2
    prefix_sum_bf16_phase2_kernel_optimized<<<1, threads_per_block>>>(
        d_block_sums, d_block_scanned, blocks_per_grid);
    // Phase 3
    prefix_sum_bf16_phase3_kernel_optimized<<<blocks_per_grid, threads_per_block>>>(
        d_temp, d_block_scanned, output, N);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_temp);
    cudaFree(d_block_sums);
    cudaFree(d_block_scanned);
}
