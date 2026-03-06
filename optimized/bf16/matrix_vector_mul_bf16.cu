#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TILE_SIZE 256
#define WARP_SIZE 32

// Optimized kernel with double-buffered shared-x prefetch and strict sequential accumulation in thread 0
__global__ void matrix_vector_mul_bf16_kernel_optimized(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int M,
    int N) {
    extern __shared__ __nv_bfloat16 sh_mem[];
    // two buffers of TILE_SIZE each
    __nv_bfloat16* sh_x_cur = sh_mem;
    __nv_bfloat16* sh_x_nxt = sh_mem + TILE_SIZE;

    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= M) return;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    const int tilePerLane = TILE_SIZE / WARP_SIZE; // =8

    int cur = 0, nxt = 1;

    // ---- preload tile 0 of x ----
    int baseIdx0 = 0;
    for (int k = 0; k < tilePerLane; ++k) {
        int idx = baseIdx0 + tid * tilePerLane + k;
        if (idx < N) {
            sh_x_cur[tid * tilePerLane + k] = x[idx];
        } else {
            sh_x_cur[tid * tilePerLane + k] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Only thread 0 does the dot-product in strict column-major order
    float sum = 0.0f;

    for (int t = 0; t < numTiles; ++t) {
        // prefetch next tile into the "nxt" buffer
        if (t + 1 < numTiles) {
            int nextBase = (t + 1) * TILE_SIZE;
            for (int k = 0; k < tilePerLane; ++k) {
                int idx = nextBase + tid * tilePerLane + k;
                if (idx < N) {
                    sh_x_nxt[tid * tilePerLane + k] = x[idx];
                } else {
                    sh_x_nxt[tid * tilePerLane + k] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            int baseA = row * N + t * TILE_SIZE;
            int remaining = N - t * TILE_SIZE;
            int len = remaining < TILE_SIZE ? remaining : TILE_SIZE;
            for (int k = 0; k < len; ++k) {
                float a = __bfloat162float(A[baseA + k]);
                float b = __bfloat162float(sh_x_cur[k]);
                sum += a * b;
            }
        }

        // swap buffers
        cur ^= 1;
        nxt ^= 1;
        if (cur == 0) {
            sh_x_cur = sh_mem;
            sh_x_nxt = sh_mem + TILE_SIZE;
        } else {
            sh_x_cur = sh_mem + TILE_SIZE;
            sh_x_nxt = sh_mem;
        }
        __syncthreads();
    }

    // write out result
    if (tid == 0) {
        y[row] = __float2bfloat16(sum);
    }
}

// Host-side wrapper remains unchanged
extern "C" void matrix_vector_mul_bf16_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int M,
    int N) {
    dim3 block(WARP_SIZE);
    dim3 grid(M);
    // Allocate shared memory for 2 * TILE_SIZE bf16 elements
    size_t shared_bytes = 2 * TILE_SIZE * sizeof(__nv_bfloat16);
    matrix_vector_mul_bf16_kernel_optimized<<<grid, block, shared_bytes>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
