#include <cuda.h>
#include <cuda_runtime.h>

// Fused kernel: handles both float4-vectorized and scalar tail in one launch
__global__ void matrix_scalar_mul_kernel_optimized(
    const float* A,
    float* B,
    float scalar,
    int M,
    int N4,
    int rem,
    int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        if (col < N4) {
            // process 4 floats at once
            const float4* A4 = reinterpret_cast<const float4*>(A);
            float4* B4 = reinterpret_cast<float4*>(B);
            int idx4 = row * N4 + col;
            float4 v = A4[idx4];
            v.x *= scalar;
            v.y *= scalar;
            v.z *= scalar;
            v.w *= scalar;
            B4[idx4] = v;
        } else if (col < N4 + rem) {
            // process leftover scalars
            int j = col - N4;
            int idx = row * N + N4 * 4 + j;
            B[idx] = A[idx] * scalar;
        }
    }
}

extern "C" void matrix_scalar_mul_optimized(
    const float* A,
    float* B,
    float scalar,
    int M,
    int N) {
    // number of full float4 vectors per row
    int N4  = N / 4;
    int rem = N - N4 * 4;

    dim3 block(32, 8);
    int totalCols = N4 + rem;
    dim3 grid(
        (totalCols + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    matrix_scalar_mul_kernel_optimized<<<grid, block>>>(
        A, B, scalar, M, N4, rem, N);
    cudaDeviceSynchronize();
}