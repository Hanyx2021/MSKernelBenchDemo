#include <cuda_runtime.h>
#include <math.h>

#define UNROLL_FACTOR 4
#define TB 16
#define TC 16

// Configuration structure for optimized high-order contraction
struct HighOrderContractionOptimizedConfig {
    int tb;                // tile size in b-dimension
    int tc;                // tile size in c-dimension
    int grid_x, grid_y, grid_z;
    int block_x, block_y;
    size_t shared_mem_bytes;
};

// Shared-memory tiled and unrolled kernel
extern "C" __global__ void high_order_contraction_kernel_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim)
{
    // 3D grid over (c_tiles, b_tiles, a_dim)
    int a = blockIdx.z;
    int b = blockIdx.y * TB + threadIdx.x;
    int c = blockIdx.x * TC + threadIdx.y;
    if (a >= a_dim || b >= b_dim || c >= c_dim) return;

    // dynamic shared memory partition
    extern __shared__ float shared_mem[];
    float* shA = shared_mem;                         // [TB][y_dim]
    float* shB = shared_mem + TB * y_dim;            // [TC][y_dim]

    float sum = 0.0f;

    // loop over x dimension, load tiles into shared memory
    for (int x = 0; x < x_dim; ++x) {
        // load A-tile: each thread with (tx, ty) loads elements of one row of A_x
        for (int yy = threadIdx.y; yy < y_dim; yy += blockDim.y) {
            int idxA = ((a * x_dim + x) * b_dim + b) * y_dim + yy;
            shA[threadIdx.x * y_dim + yy] = A[idxA];
        }
        // load B-tile: each thread with (tx, ty) loads elements of one row of B_x
        for (int yy = threadIdx.x; yy < y_dim; yy += blockDim.x) {
            int idxB = ((x * c_dim + c) * y_dim) + yy;
            shB[threadIdx.y * y_dim + yy] = B[idxB];
        }

        __syncthreads();

        // pointers to this thread's row in shared memory
        float* rowA = shA + threadIdx.x * y_dim;
        float* rowB = shB + threadIdx.y * y_dim;

        int limit = y_dim - (UNROLL_FACTOR - 1);
        int yy = 0;
        // unrolled FMA
        for (; yy < limit; yy += UNROLL_FACTOR) {
            sum = fmaf(rowA[yy + 0], rowB[yy + 0], sum);
            sum = fmaf(rowA[yy + 1], rowB[yy + 1], sum);
            sum = fmaf(rowA[yy + 2], rowB[yy + 2], sum);
            sum = fmaf(rowA[yy + 3], rowB[yy + 3], sum);
        }
        // tail
        for (; yy < y_dim; ++yy) {
            sum = fmaf(rowA[yy], rowB[yy], sum);
        }

        __syncthreads();
    }

    // write result
    int idxC = ((a * b_dim + b) * c_dim) + c;
    C[idxC] = sum;
}

// External C wrapper for the optimized operator
extern "C" void high_order_contraction_optimized(
    const float* A,
    const float* B,
    float* C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim)
{
    // tile sizes
    const int tb = TB;
    const int tc = TC;

    // compute grid dimensions
    int grid_x = (c_dim + tc - 1) / tc;
    int grid_y = (b_dim + tb - 1) / tb;
    int grid_z = a_dim;

    dim3 block(tb, tc);
    dim3 grid(grid_x, grid_y, grid_z);

    // shared memory size: two tiles of shape [tile_dim][y_dim]
    size_t shared_mem = (size_t)(tb + tc) * y_dim * sizeof(float);

    // optional config (for tuning/logging)
    HighOrderContractionOptimizedConfig config = {tb, tc, grid_x, grid_y, grid_z,
                                                  tb, tc, shared_mem};

    // launch optimized kernel
    high_order_contraction_kernel_optimized<<<grid, block, shared_mem>>>(
        A, B, C,
        a_dim, b_dim, c_dim,
        x_dim, y_dim
    );
    cudaDeviceSynchronize();
}