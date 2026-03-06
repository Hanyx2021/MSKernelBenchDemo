#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Vectorized 2x bf16 per thread copy kernel
__global__ void matrix_copy_bf16_kernel_optimized(const __nv_bfloat16* A,
                                                  __nv_bfloat16* B,
                                                  int M,
                                                  int N) {
    size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_vec = (total + 1) / 2;

    if (idx < total_vec) {
        // Reinterpret pointers as 32-bit to copy two bf16s at once
        const uint32_t* A32 = reinterpret_cast<const uint32_t*>(A);
        uint32_t*       B32 = reinterpret_cast<uint32_t*>(B);

        // Handle odd tail: last thread and odd total
        if ((idx == total_vec - 1) && (total & 1)) {
            size_t pos = total - 1;
            B[pos] = A[pos];
        } else {
            // Vectorized copy: two bf16 elements per uint32 load/store
            B32[idx] = A32[idx];
        }
    }
}

// External C wrapper for the optimized kernel
extern "C" void matrix_copy_bf16_optimized(const __nv_bfloat16* A,
                                            __nv_bfloat16* B,
                                            int M,
                                            int N) {
    size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
    size_t total_vec = (total + 1) / 2;
    const int threads = 256;
    int blocks = static_cast<int>((total_vec + threads - 1) / threads);

    matrix_copy_bf16_kernel_optimized<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}