#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimension for shared-memory tiling
#define TILE_DIM 32

// Optimized matrix multiplication kernel using shared-memory tiling and register blocking
__global__ void matrix_mul_kernel_optimized(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N) {
    // Tile indices
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Shared memory tiles for A and B
    __shared__ float Asub[TILE_DIM][TILE_DIM];
    __shared__ float Bsub[TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_DIM + threadIdx.x;
        int bRow = t * TILE_DIM + threadIdx.y;

        // Load A tile element or zero if out of bounds
        if (row < N && aCol < N) {
            Asub[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Load B tile element or zero if out of bounds
        if (bRow < N && col < N) {
            Bsub[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute partial dot-product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum = fmaf(Asub[threadIdx.y][k], Bsub[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    // Write the result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void matrix_power_optimized(
    const float* A,
    float* B,
    int N,
    int P) {
    // Allocate device buffers
    size_t size = static_cast<size_t>(N) * N * sizeof(float);
    float* d_A = nullptr;
    float* d_R = nullptr;
    float* d_temp = nullptr;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_R, size);
    cudaMalloc(&d_temp, size);

    // Copy input matrix A to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Configure execution parameters
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (N + TILE_DIM - 1) / TILE_DIM);

    // Compute initial square (A^2)
    matrix_mul_kernel_optimized<<<grid, block>>>(d_A, d_A, d_R, N);

    // Multiply by A (P-2) more times to get A^P
    for (int i = 2; i < P; ++i) {
        matrix_mul_kernel_optimized<<<grid, block>>>(d_R, d_A, d_temp, N);
        // Swap buffers
        float* swap_ptr = d_R;
        d_R    = d_temp;
        d_temp = swap_ptr;
    }

    // Synchronize to ensure all kernels are done
    cudaDeviceSynchronize();

    // Copy result (A^P) back to host
    cudaMemcpy(B, d_R, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_R);
    cudaFree(d_temp);
}
