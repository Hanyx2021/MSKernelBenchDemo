#include <cuda_runtime.h>
#include <cuda.h>

// Configuration struct for optimized merge sort
struct MergeSortConfigOptimized {
    int threadsPerBlock;
};

// Default configuration: tune threadsPerBlock in [128,256,512,1024]
static const MergeSortConfigOptimized cfg = { 256 };

// Device binary search split for merge-path
__device__ int binary_search_split_optimized(const float* A, int lenA, const float* B, int lenB, int k) {
    int low = (k - lenB > 0) ? (k - lenB) : 0;
    int high = (k < lenA) ? k : lenA;
    while (low < high) {
        int mid = (low + high) >> 1;
        if (A[mid] < B[k - mid - 1]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Multi-block parallel merge kernel optimized
__global__ void merge_sort_multiblock_kernel_optimized(
    const float* src, float* dst, int N, int blockSize, int blocksPerRun) {
    int globalBlockId = blockIdx.x;
    int runId      = globalBlockId / blocksPerRun;
    int subBlockId = globalBlockId % blocksPerRun;
    int start = runId * 2 * blockSize;
    if (start >= N) return;

    // Compute lengths of the two runs
    int lenA = (N - start < blockSize) ? (N - start) : blockSize;
    int rem  = N - start - blockSize;
    int lenB = (rem > 0) ? ((rem < blockSize) ? rem : blockSize) : 0;
    int L = lenA + lenB;
    if (L <= 0) return;

    // Compute this run-pair block's output segment [k_start, k_end)
    int perRun = L / blocksPerRun;
    int remRun = L % blocksPerRun;
    int k_start = (subBlockId < remRun)
                  ? subBlockId * (perRun + 1)
                  : remRun * (perRun + 1) + (subBlockId - remRun) * perRun;
    int myLen = (subBlockId < remRun) ? (perRun + 1) : perRun;
    if (myLen <= 0) return;
    int k_end = k_start + myLen;

    // If there's nothing to merge (only A), copy subrange
    if (lenB <= 0) {
        // Each thread copies its portion
        int t = threadIdx.x;
        int T = blockDim.x;
        for (int idx = k_start + t; idx < k_end; idx += T) {
            dst[start + idx] = src[start + idx];
        }
        return;
    }

    // Split work among threads within block
    int T = blockDim.x;
    int t = threadIdx.x;
    int perThread = myLen / T;
    int remThread = myLen % T;
    int ts_k_start = k_start + ((t < remThread)
                       ? t * (perThread + 1)
                       : remThread * (perThread + 1) + (t - remThread) * perThread);
    int ts_len = (t < remThread) ? (perThread + 1) : perThread;
    if (ts_len <= 0) return;
    int ts_k_end = ts_k_start + ts_len;

    // Pointers to runs
    const float* A = src + start;
    const float* B = src + start + blockSize;

    // Locate split points for the thread's segment
    int pA_start = binary_search_split_optimized(A, lenA, B, lenB, ts_k_start);
    int pB_start = ts_k_start - pA_start;
    int pA_end   = binary_search_split_optimized(A, lenA, B, lenB, ts_k_end);
    int pB_end   = ts_k_end   - pA_end;

    // Merge subranges
    int i = pA_start;
    int j = pB_start;
    int out = start + ts_k_start;
    while (i < pA_end && j < pB_end) {
        float va = A[i];
        float vb = B[j];
        if (va <= vb) {
            dst[out++] = va;
            ++i;
        } else {
            dst[out++] = vb;
            ++j;
        }
    }
    while (i < pA_end) dst[out++] = A[i++];
    while (j < pB_end) dst[out++] = B[j++];
}

// External C wrapper: optimized sorting
extern "C" void sorting_optimized(float* data, int N) {
    // Allocate temporary buffer
    float* d_temp = nullptr;
    cudaMalloc(&d_temp, N * sizeof(float));

    const float* src = data;
    float* dst = d_temp;

    int threadsPerBlock = cfg.threadsPerBlock;
    // Target total blocks for GPU occupancy (e.g., 4x SM count)
    const int targetTotalBlocks = 512;

    // Merge steps with ping-pong buffers
    for (int blockSize = 1; blockSize < N; blockSize <<= 1) {
        int R = (N + 2 * blockSize - 1) / (2 * blockSize);
        int blocksPerRun = 1;
        if (R > 0) {
            blocksPerRun = (targetTotalBlocks + R - 1) / R;
            if (blocksPerRun < 1) blocksPerRun = 1;
        }
        int totalBlocks = R * blocksPerRun;

        merge_sort_multiblock_kernel_optimized<<<totalBlocks, threadsPerBlock>>>(
            src, dst, N, blockSize, blocksPerRun);
        // Swap source and destination
        const float* tmp = src;
        src = dst;
        dst = (float*)tmp;
    }

    // If final data is in temporary buffer, copy back
    if (src != data) {
        cudaMemcpy(data, src, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Clean up
    cudaFree(d_temp);
    cudaDeviceSynchronize();
}