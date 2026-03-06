#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Optimized kernel: 1-to-1 mapping of threads to elements, no strided loop
__global__ void vector_add_bf16_kernel_optimized(const __nv_bfloat16* A,
                                                 const __nv_bfloat16* B,
                                                 __nv_bfloat16* C,
                                                 int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Convert bfloat16 to float, add, then convert back
        float a = __bfloat162float(A[idx]);
        float b = __bfloat162float(B[idx]);
        C[idx]   = __float2bfloat16(a + b);
    }
}

// External C wrapper with optimized launch configuration
extern "C" void vector_add_bf16_optimized(const __nv_bfloat16* A,
                                            const __nv_bfloat16* B,
                                            __nv_bfloat16* C,
                                            int N) {
    const int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    dim3 block(block_size, 1, 1);
    dim3 grid(grid_size, 1, 1);
    
    vector_add_bf16_kernel_optimized<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();
}