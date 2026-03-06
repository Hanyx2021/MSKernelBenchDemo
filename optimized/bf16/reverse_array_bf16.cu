#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Optimized kernel using strided loop work distribution to improve memory-latency hiding
__global__ void reverse_array_bf16_kernel_optimized(__nv_bfloat16* output, __nv_bfloat16* input, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int half = (N + 1) / 2;  // include middle element for odd N

    // Strided loop: each thread processes multiple swap pairs
    for (int i = gid; i < half; i += total_threads) {
        int j = N - 1 - i;
        __nv_bfloat16 a = input[i];
        __nv_bfloat16 b = input[j];
        output[i] = b;
        output[j] = a;
    }
}

extern "C" void reverse_array_bf16_optimized(__nv_bfloat16* output, __nv_bfloat16* input, int N) {
    // Compute number of pairs to swap (including middle element if N is odd)
    int half = (N + 1) / 2;
    // Launch configuration
    const int threads_per_block = 256;
    int blocks = (half + threads_per_block - 1) / threads_per_block;

    reverse_array_bf16_kernel_optimized<<<blocks, threads_per_block>>>(output, input, N);
    cudaDeviceSynchronize();
}