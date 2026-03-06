#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions and stripe height
#define TILE_DIM 32
#define BLOCK_ROWS 8

// Optimized tiled transpose kernel using cp.async pipeline and double buffering
__global__ void matrix_transpose_kernel_optimized(
    const float* __restrict__ A,
    float* __restrict__ B,
    int M,
    int N) {
#if (__CUDA_ARCH__ >= 890)
    // Double buffer in shared memory with padding to avoid bank conflicts
    extern __shared__ float shared_mem[];
    float (*tile_buf)[TILE_DIM + 1] = (float (*)[TILE_DIM + 1])shared_mem;
    float (*tile_buf2)[TILE_DIM + 1] = (float (*)[TILE_DIM + 1])(shared_mem + (TILE_DIM * (TILE_DIM + 1)));

    const float* src_ptr = A;
    float* dst_ptr = B;

    // Calculate tile origin
    int block_x = blockIdx.x * TILE_DIM;
    int block_y = blockIdx.y * TILE_DIM;
    int xIndex = block_x + threadIdx.x;
    int yIndex = block_y + threadIdx.y;

    int stripes = (TILE_DIM + BLOCK_ROWS - 1) / BLOCK_ROWS;

    // Pipeline: load first stripe into buffer 0
    int buf_idx = 0;
    for (int i = 0; i < BLOCK_ROWS; ++i) {
        int y = yIndex + i;
        int x = xIndex;
        if (x < N && y < M) {
            // Issue cp.async load: one element at a time
            asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;"
                          :
                          : "r"(&tile_buf[threadIdx.y + i][threadIdx.x]),
                            "l"(src_ptr + (y * N + x)),
                            "n"(sizeof(float))
                          :);
        }
    }
    asm volatile("cp.async.commit_group;\n");

    // Process pipeline stages
    for (int s = 1; s < stripes; ++s) {
        // Wait for previous commit to complete for buffer 0
        asm volatile("cp.async.wait_group %0;\n" :: "n"(1));

        // Launch loading for stripe s into the alternate buffer
        int cur_buf = (buf_idx ^ 1);
        for (int i = 0; i < BLOCK_ROWS; ++i) {
            int y = yIndex + s * BLOCK_ROWS + i;
            int x = xIndex;
            if (x < N && y < M) {
                asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;"
                              :
                              : "r"(&tile_buf2[threadIdx.y + i][threadIdx.x]),
                                "l"(src_ptr + (y * N + x)),
                                "n"(sizeof(float))
                              :);
            }
        }
        asm volatile("cp.async.commit_group;\n");

        // Transpose & store buffer buf_idx (previous stripe)
        int ty = block_x + threadIdx.y;
        int tx = block_y + buf_idx * BLOCK_ROWS + threadIdx.x;
        for (int i = 0; i < BLOCK_ROWS; ++i) {
            float val = tile_buf[threadIdx.x][threadIdx.y + i];
            int row = tx + i;
            int col = ty;
            if (row < N && col < M) {
                dst_ptr[row * M + col] = val;
            }
        }

        // swap buffers
        buf_idx ^= 1;
        // swap pointers for convenience
        float (*tmp)[TILE_DIM + 1] = tile_buf;
        tile_buf = tile_buf2;
        tile_buf2 = tmp;
    }

    // Final stripe: wait and process last loaded stripe
    asm volatile("cp.async.wait_group %0;\n" :: "n"(1));
    // Transpose & store last stripe
    for (int i = 0; i < BLOCK_ROWS; ++i) {
        float val = tile_buf[threadIdx.x][threadIdx.y + i];
        int row = block_y + buf_idx * BLOCK_ROWS + i;
        int col = block_x + threadIdx.x;
        if ((block_x + threadIdx.x) < N && row < M) {
            B[row * M + col] = val;
        }
    }

    // Ensure all async operations are done
    asm volatile("cp.async.wait_all;\n");
#else
    // Fallback for older architectures

    // compute tile origin exactly as above
    int block_x = blockIdx.x * TILE_DIM;
    int block_y = blockIdx.y * TILE_DIM;

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    // Phase 1: Load
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = block_y + threadIdx.y + i;
        int x = block_x + threadIdx.x;
        if (x < N && y < M) {
            tile[threadIdx.y + i][threadIdx.x] = A[y * N + x];
        }
    }
    __syncthreads();
    // Phase 2: Store
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = block_x + threadIdx.y + i;
        int x = block_y + threadIdx.x;
        if (x < M && y < N) {
            B[y * M + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
#endif
}

extern "C" void matrix_transpose_optimized(
    const float* A,
    float* B,
    int M,
    int N) {
    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (M + TILE_DIM - 1) / TILE_DIM,
              1);
    // Calculate shared memory size: two buffers
    size_t shmem_size = 2 * TILE_DIM * (TILE_DIM + 1) * sizeof(float);

    matrix_transpose_kernel_optimized<<<grid, block, shmem_size>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
