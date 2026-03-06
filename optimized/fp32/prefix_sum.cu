#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel 1: Warp‐Shuffle Hierarchical Scan (exclusive-to-inclusive transform inside)
__global__ void prefix_sum_intra_block_scan_kernel_optimized(
    float* output,
    const float* input,
    float* blockSums,
    int N)
{
    // Each block handles segment of 2 * blockDim.x elements
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int segSize = threads * 2;
    const int blockStart = blockIdx.x * segSize;

    // Compute warp and lane IDs
    const int lane = tid & 31;
    const int warpId = tid >> 5;
    const int warpCount = (threads + 31) / 32;

    // Load elements (pad with zero if out of bounds)
    int idx1 = blockStart + 2 * tid;
    int idx2 = blockStart + 2 * tid + 1;
    float x0 = (idx1 < N) ? input[idx1] : 0.0f;
    float x1 = (idx2 < N) ? input[idx2] : 0.0f;

    // Compute lane sum (two elements)
    float laneSum = x0 + x1;

    // Intra-warp exclusive scan on laneSum via shuffle
    unsigned int fullMask = 0xFFFFFFFF;
    float sum = laneSum;
    // Inclusive scan in registers
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float t = __shfl_up_sync(fullMask, sum, offset);
        if (lane >= offset) sum += t;
    }
    // sum is inclusive scan of laneSum across warp (sum of laneSum[0..lane])
    float inclusiveLaneSum = sum;
    float exclusiveLaneSum = inclusiveLaneSum - laneSum;

    // Compute per-element partial results (inclusive)
    float out0 = exclusiveLaneSum + x0;
    float out1 = inclusiveLaneSum;

    // Shared memory for inter-warp prefix sums
    extern __shared__ float shm[];  // size >= warpCount
    float* warpSums = shm;         // to hold inclusive warp sums

    // Last lane of each warp writes its inclusive sum to shared
    if (lane == 31) {
        warpSums[warpId] = inclusiveLaneSum;
    }
    __syncthreads();

    // One thread (thread 0) does small exclusive scan on warpSums
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < warpCount; ++i) {
            float temp = warpSums[i];
            warpSums[i] = s;
            s += temp;
        }
        // Write block total to blockSums
        blockSums[blockIdx.x] = s;
    }
    __syncthreads();

    // Broadcast warp offsets and add to partial results
    float warpOffset = warpSums[warpId];
    out0 += warpOffset;
    out1 += warpOffset;

    // Write results back (inclusive scan values)
    if (idx1 < N) output[idx1] = out0;
    if (idx2 < N) output[idx2] = out1;
}

// Kernel 3: Add block offsets to each element (unchanged)
__global__ void prefix_sum_add_block_offsets_kernel_optimized(
    float* output,
    const float* blockOffsets,
    int N)
{
    int tid = threadIdx.x;
    int threads = blockDim.x;
    int segSize = threads * 2;
    int blockStart = blockIdx.x * segSize;
    float offset = blockOffsets[blockIdx.x];

    int idx1 = blockStart + 2 * tid;
    int idx2 = blockStart + 2 * tid + 1;
    if (idx1 < N) output[idx1] += offset;
    if (idx2 < N) output[idx2] += offset;
}

// External C wrapper: prefix_sum_optimized
extern "C" void prefix_sum_optimized(
    float* output,
    const float* input,
    const int N)
{
    const int threads = 256;
    const int segSize = threads * 2;
    int blocks = (N + segSize - 1) / segSize;
    int warpCount = (threads + 31) / 32;
    size_t sharedMemSize = warpCount * sizeof(float);

    // Allocate temporary arrays for block sums and offsets
    float* d_blockSums = nullptr;
    float* d_blockOffsets = nullptr;
    cudaMalloc(&d_blockSums, blocks * sizeof(float));
    cudaMalloc(&d_blockOffsets, blocks * sizeof(float));

    // Kernel 1: hierarchical prefix-scan and compute block sums
    prefix_sum_intra_block_scan_kernel_optimized<<<blocks, threads, sharedMemSize>>>(
        output, input, d_blockSums, N);
    cudaDeviceSynchronize();

    // Host-side prefix sum of block sums (small array)
    float* h_blockSums = (float*)malloc(blocks * sizeof(float));
    float* h_blockOffsets = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_blockSums, d_blockSums, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    h_blockOffsets[0] = 0.0f;
    for (int i = 1; i < blocks; ++i) {
        h_blockOffsets[i] = h_blockOffsets[i - 1] + h_blockSums[i - 1];
    }
    cudaMemcpy(d_blockOffsets, h_blockOffsets, blocks * sizeof(float), cudaMemcpyHostToDevice);
    free(h_blockSums);
    free(h_blockOffsets);

    // Kernel 3: add block offsets
    prefix_sum_add_block_offsets_kernel_optimized<<<blocks, threads>>>(
        output, d_blockOffsets, N);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_blockSums);
    cudaFree(d_blockOffsets);
}
