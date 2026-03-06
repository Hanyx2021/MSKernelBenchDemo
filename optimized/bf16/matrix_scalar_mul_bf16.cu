#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Vectorized kernel: each thread processes 4 bf16 elements packed in a 64-bit word
__global__ void matrix_scalar_mul_bf16_kernel_optimized(
    const __nv_bfloat16* A, __nv_bfloat16* B,
    __nv_bfloat16 scalar, int M, int N, int N_groups) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_group = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col_group >= N_groups) return;

    // reinterpret A and B as arrays of packed 4 bf16 (=64-bit)
    const uint64_t* A_pack = reinterpret_cast<const uint64_t*>(A);
    uint64_t* B_pack = reinterpret_cast<uint64_t*>(B);

    // load packed data
    uint64_t data = A_pack[row * N_groups + col_group];
    // unpack, multiply, repack
    union { uint64_t u; __nv_bfloat16 v[4]; } tmp;
    tmp.u = data;
    float s = __bfloat162float(scalar);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float x = __bfloat162float(tmp.v[i]);
        tmp.v[i] = __float2bfloat16(x * s);
    }
    // store result
    B_pack[row * N_groups + col_group] = tmp.u;
}

// Tail kernel: handle remaining columns when N % 4 != 0
__global__ void matrix_scalar_mul_bf16_tail_kernel_optimized(
    const __nv_bfloat16* A, __nv_bfloat16* B,
    __nv_bfloat16 scalar, int M, int N, int N_groups, int rem) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= rem) return;

    int idx = row * N + N_groups * 4 + col;
    float x = __bfloat162float(A[idx]);
    float prod = x * __bfloat162float(scalar);
    B[idx] = __float2bfloat16(prod);
}

extern "C" void matrix_scalar_mul_bf16_optimized(
    const __nv_bfloat16* A, __nv_bfloat16* B,
    __nv_bfloat16 scalar, int M, int N) {
    // Compute number of full 4-element groups and remainder
    int N_groups = N / 4;
    int rem = N - N_groups * 4;

    // Launch vectorized kernel for groups of 4 bf16
    if (N_groups > 0) {
        dim3 block(128, 4, 1);
        dim3 grid((N_groups + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y,
                  1);
        matrix_scalar_mul_bf16_kernel_optimized<<<grid, block>>>(
            A, B, scalar, M, N, N_groups);
    }

    // Launch tail kernel for remaining columns
    if (rem > 0) {
        dim3 blockTail(128, 4, 1);
        dim3 gridTail((rem + blockTail.x - 1) / blockTail.x,
                       (M + blockTail.y - 1) / blockTail.y,
                       1);
        matrix_scalar_mul_bf16_tail_kernel_optimized<<<gridTail, blockTail>>>(
            A, B, scalar, M, N, N_groups, rem);
    }

    cudaDeviceSynchronize();
}
