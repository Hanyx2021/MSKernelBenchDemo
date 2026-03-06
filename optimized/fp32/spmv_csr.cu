#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_COLS 2048

// Optimized kernel: cache entire x vector in shared memory of size NUM_COLS
__global__ void spmv_csr_kernel_optimized(
    int rows,
    const float* __restrict__ values,
    const int*   __restrict__ col_indices,
    const int*   __restrict__ row_offsets,
    const float* __restrict__ x,
    float*       __restrict__ y)
{
    extern __shared__ float s_x[];  // dynamic shared memory for x
    int tid = threadIdx.x;

    // Load exactly NUM_COLS elements of x into shared memory
    for (int i = tid; i < NUM_COLS; i += blockDim.x) {
        s_x[i] = __ldg(&x[i]);
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.x + tid;
    if (row < rows) {
        float sum = 0.0f;
        int start = row_offsets[row];
        int end   = row_offsets[row + 1];
        for (int j = start; j < end; ++j) {
            int c = col_indices[j];  // guaranteed 0 <= c < NUM_COLS
            sum += values[j] * s_x[c];
        }
        y[row] = sum;
    }
}

extern "C" void spmv_csr_optimized(
    int rows,
    const float* values,
    const int*   col_indices,
    const int*   row_offsets,
    const float* x,
    float*       y)
{
    // Bias cache toward L1 for better temporal reuse of x
    cudaFuncSetCacheConfig(spmv_csr_kernel_optimized, cudaFuncCachePreferL1);

    const int THREADS_PER_BLOCK = 256;
    int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid(blocks, 1, 1);

    // Allocate dynamic shared memory = NUM_COLS floats
    size_t shmem_bytes = NUM_COLS * sizeof(float);
    spmv_csr_kernel_optimized<<<grid, block, shmem_bytes>>>(
        rows, values, col_indices, row_offsets, x, y);
    cudaDeviceSynchronize();
}