#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>

// Small chunk size for initial bitonic sort
#define CHUNK 1024

// Device parameters for dynamic partitioning
#define SM_COUNT 128       // Number of SMs on target device
#define BSM 8              // Desired blocks per SM
#define TARGET_BLOCKS (SM_COUNT * BSM)

// Merge kernel thread configuration
#define MERGE_THREADS_OPT 256
#define WARP_SIZE_OPT 32

// Utility for integer division with rounding up
inline int iDivUp(int a, int b) {
    return (a + b - 1) / b;
}

// In-block bitonic sort for small chunks using shared memory (one thread per element)
__global__ void bitonic_sort_bf16_kernel_optimized(__nv_bfloat16* data, int N) {
    extern __shared__ float sdata[];      // CHUNK floats per block
    int tid = threadIdx.x;
    int base = blockIdx.x * CHUNK;
    int idx = base + tid;

    // load one element per thread
    float v = (idx < N)
              ? __bfloat162float(data[idx])
              : FLT_MAX;
    sdata[tid] = v;
    __syncthreads();

    // classic bitonic network over CHUNK elements
    for (int k = 2; k <= CHUNK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid && ixj < CHUNK) {
                bool up = ((tid & k) == 0);
                float a = sdata[tid];
                float b = sdata[ixj];
                // when up==true we want a<=b, when up==false we want a>=b
                if ((a > b) == up) {
                    sdata[tid]  = b;
                    sdata[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // write back sorted data to global memory
    if (idx < N) {
        data[idx] = __float2bfloat16(sdata[tid]);
    }
}

// Partitioned merge kernel with dynamic partitions per pass
__global__ void merge_partitioned_bf16_kernel_optimized(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int N,
    int blockSize,
    int partitions)
{
    int global_block = blockIdx.x;
    int part_idx = global_block % partitions;
    int run_idx = global_block / partitions;

    int start = run_idx * 2 * blockSize;
    if (start >= N) return;
    int mid = start + blockSize;
    mid = (mid < N) ? mid : N;
    int end = start + 2 * blockSize;
    end = (end < N) ? end : N;

    int A_size = mid - start;
    int B_size = end - mid;
    int total = A_size + B_size;

    // Partition length
    int L = (total + partitions - 1) / partitions;
    int local_start = part_idx * L;
    if (local_start >= total) return;
    int local_L = (local_start + L < total) ? L : (total - local_start);

    int tid = threadIdx.x;
    if (tid >= local_L) return;

    int k = local_start + tid;
    // Merge path binary search to find split
    int low = k - B_size;
    if (low < 0) low = 0;
    int high = (k < A_size) ? k : A_size;
    while (low < high) {
        int midp = (low + high) >> 1;
        float aval = __bfloat162float(input[start + midp]);
        int bidx = k - midp - 1;
        float bval = FLT_MAX;
        if (bidx >= 0 && bidx < B_size) {
            bval = __bfloat162float(input[mid + bidx]);
        }
        if (aval <= bval) {
            low = midp + 1;
        } else {
            high = midp;
        }
    }
    int a_j = low;
    int b_j = k - a_j;

    float valA = FLT_MAX;
    if (a_j < A_size) {
        valA = __bfloat162float(input[start + a_j]);
    }
    float valB = FLT_MAX;
    if (b_j < B_size) {
        valB = __bfloat162float(input[mid + b_j]);
    }
    float v = (valA <= valB) ? valA : valB;

    output[start + k] = __float2bfloat16(v);
}

extern "C" void sorting_bf16_optimized(__nv_bfloat16* data, int N) {
    // Allocate temporary buffer
    __nv_bfloat16* d_temp;
    cudaMalloc(&d_temp, N * sizeof(__nv_bfloat16));
    cudaMemcpy(d_temp, data, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

    // 1) In-block bitonic sort on chunks
    int blocksChunk = iDivUp(N, CHUNK);
    bitonic_sort_bf16_kernel_optimized<<<blocksChunk, CHUNK, CHUNK * sizeof(float)>>>(data, N);

    // Copy sorted chunks to temp buffer
    cudaMemcpy(d_temp, data, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

    // 2) Global merge passes with dynamic partitions (ping-pong)
    const __nv_bfloat16* in = d_temp;
    __nv_bfloat16* out = data;
    for (int blockSize = CHUNK; blockSize < N; blockSize <<= 1) {
        int runPairs = iDivUp(N, 2 * blockSize);
        int partitions = iDivUp(TARGET_BLOCKS, runPairs);
        if (partitions < 1) partitions = 1;
        int maxAllowed = (2 * blockSize + WARP_SIZE_OPT - 1) / WARP_SIZE_OPT;
        if (maxAllowed < 1) maxAllowed = 1;
        if (partitions > maxAllowed) partitions = maxAllowed;

        int grid = runPairs * partitions;
        merge_partitioned_bf16_kernel_optimized<<<grid, MERGE_THREADS_OPT>>>(in, out, N, blockSize, partitions);

        // swap buffers
        const __nv_bfloat16* tmp = in;
        in = out;
        out = const_cast<__nv_bfloat16*>(tmp);
    }

    // If final result is not in "data", copy it back
    if (in != data) {
        cudaMemcpy(data, in, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    }
    cudaFree(d_temp);
    cudaDeviceSynchronize();
}