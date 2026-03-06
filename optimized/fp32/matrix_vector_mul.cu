#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel with dynamic tile size based on blockDim.x
__global__ void matrix_vector_mul_kernel_optimized(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N) {
    extern __shared__ float x_tile[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Use blockDim.x as tile size
    int tile_size = blockDim.x;

    // Loop over tiles of the input vector x
    for (int c = 0; c < N; c += tile_size) {
        int rem = N - c;
        int curr_tile_len = rem < tile_size ? rem : tile_size;

        // Load a tile of x into shared memory
        if (threadIdx.x < curr_tile_len) {
            x_tile[threadIdx.x] = x[c + threadIdx.x];
        }
        __syncthreads();

        // Perform partial dot-product for this tile
        if (row < M) {
            int base = row * N + c;
            for (int j = 0; j < curr_tile_len; ++j) {
                sum += A[base + j] * x_tile[j];
            }
        }
        __syncthreads();
    }

    // Write the final result
    if (row < M) {
        y[row] = sum;
    }
}

// External C wrapper launching the optimized kernel
extern "C" void matrix_vector_mul_optimized(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N) {
    // Tune block size for higher occupancy (default to 64 threads per block)
    int block_size = (M < 64) ? M : 64;
    int grid_size = (M + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * sizeof(float);

    // Kernel launch with dynamic shared memory for x_tile
    matrix_vector_mul_kernel_optimized<<<grid_size, block_size, shared_mem_size>>>(
        A, x, y, M, N);
    cudaDeviceSynchronize();
}