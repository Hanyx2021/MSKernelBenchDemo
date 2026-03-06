#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define MAX_KERNEL_SIZE 32

// Kernel weights in constant memory
__constant__ float c_kernel_const[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// Optimized 2D convolution kernel using shared-memory tiling and cached constant memory
__global__ void conv_2d_kernel_optimized(const float* input, float* output,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    // Compute output dimensions
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    // Tile origin in input
    int tile_start_x = blockIdx.x * BLOCK_X;
    int tile_start_y = blockIdx.y * BLOCK_Y;

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global output coordinates
    int out_x = tile_start_x + tx;
    int out_y = tile_start_y + ty;

    // Shared memory tile dimensions (including halo)
    int tile_width = BLOCK_X + kernel_cols - 1;
    int tile_height = BLOCK_Y + kernel_rows - 1;

    extern __shared__ float smem[];

    // Load input tile (with halo) into shared memory
    for (int y = ty; y < tile_height; y += BLOCK_Y) {
        for (int x = tx; x < tile_width; x += BLOCK_X) {
            int in_x = tile_start_x + x;
            int in_y = tile_start_y + y;
            float val = (in_x < input_cols && in_y < input_rows) ?
                input[in_y * input_cols + in_x] : 0.0f;
            smem[y * tile_width + x] = val;
        }
    }
    __syncthreads();

    // Perform convolution using shared memory for input
    if (out_x < output_cols && out_y < output_rows) {
        float sum = 0.0f;
        #pragma unroll
        for (int m = 0; m < kernel_rows; ++m) {
            #pragma unroll
            for (int n = 0; n < kernel_cols; ++n) {
                float in_val = smem[(ty + m) * tile_width + (tx + n)];
                float ker_val = c_kernel_const[m * kernel_cols + n];
                sum += in_val * ker_val;
            }
        }
        output[out_y * output_cols + out_x] = sum;
    }
}

// External C wrapper for the optimized convolution
extern "C" void conv_2d_optimized(const float* input, const float* kernel, float* output,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    // Copy kernel weights to constant memory
    size_t kernel_size_bytes = kernel_rows * kernel_cols * sizeof(float);
    cudaMemcpyToSymbol(c_kernel_const, kernel, kernel_size_bytes, 0, cudaMemcpyHostToDevice);

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((output_cols + BLOCK_X - 1) / BLOCK_X,
              (output_rows + BLOCK_Y - 1) / BLOCK_Y);

    // Dynamic shared memory size: tile area (including halo)
    size_t shared_mem_size = sizeof(float) * (BLOCK_Y + kernel_rows - 1) * (BLOCK_X + kernel_cols - 1);

    // Launch optimized kernel
    conv_2d_kernel_optimized<<<grid, block, shared_mem_size>>>(input, output,
        input_rows, input_cols,
        kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
