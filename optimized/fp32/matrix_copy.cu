#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Vectorized kernel to copy 4 floats at a time using float4
__global__ void matrix_copy_kernel_optimized(const float* __restrict__ A,
                                             float* __restrict__ B,
                                             int M,
                                             int N4) {
    int vecIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int row    = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && vecIdx < N4) {
        // reinterpret pointers as float4 for vectorized load/store
        const float4* A4 = reinterpret_cast<const float4*>(A + row * (size_t)N4 * 4);
        float4*       B4 = reinterpret_cast<float4*>(B + row * (size_t)N4 * 4);
        B4[vecIdx] = A4[vecIdx];
    }
}

// Tail kernel to copy remaining elements when N % 4 != 0
__global__ void matrix_copy_tail_kernel_optimized(const float* __restrict__ A,
                                                  float* __restrict__ B,
                                                  int M,
                                                  int N,
                                                  int N4,
                                                  int rem) {
    int colOffset = N4 * 4;
    int row       = blockIdx.x;
    int tid       = threadIdx.x;
    if (row < M && tid < rem) {
        int col = colOffset + tid;
        B[row * N + col] = A[row * N + col];
    }
}

extern "C" void matrix_copy_optimized(float* A, float* B, int M, int N) {
    // Number of full float4 elements per row
    int N4  = N / 4;
    int rem = N - N4 * 4;

    // Launch vectorized kernel if there are any full 4-float groups
    if (N4 > 0) {
        // We choose a 2D block of 256×4 threads = 1024 threads total
        // so that grid.y = ceil(M/4) stays <= 65535 even when M = 65536.
        const int tx = 256;
        const int ty = 4;
        dim3 block(tx, ty, 1);
        dim3 grid((N4 + tx - 1) / tx,
                  (M   + ty - 1) / ty,
                  1);

        matrix_copy_kernel_optimized<<<grid, block>>>(A, B, M, N4);
    }

    // Launch tail-kernel if remainder exists
    if (rem > 0) {
        // one block per row, rem threads per block
        dim3 blockTail(rem, 1, 1);
        dim3 gridTail(M, 1, 1);
        matrix_copy_tail_kernel_optimized<<<gridTail, blockTail>>>(
            A, B, M, N, N4, rem);
    }

    cudaDeviceSynchronize();
}
