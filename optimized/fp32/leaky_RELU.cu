#include <cuda.h>
#include <cuda_runtime.h>

// Optimized leaky ReLU kernel with vectorized (float4) loads/stores and grid-stride loop
__global__ void leaky_RELU_kernel_optimized(
    float* output,
    const float* __restrict__ input,
    int N,
    float alpha) {
    // Cast to float4 pointers for vectorized access
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Number of full float4 vectors
    int n_vec = N / 4;

    // Process main vectorized part
    for (int vecIdx = idx; vecIdx < n_vec; vecIdx += stride) {
        // Load through read-only cache
        float4 in4 = __ldg(input4 + vecIdx);
        float4 out4;
        // Apply leaky ReLU to each component
        out4.x = (in4.x < 0.0f) ? (alpha * in4.x) : in4.x;
        out4.y = (in4.y < 0.0f) ? (alpha * in4.y) : in4.y;
        out4.z = (in4.z < 0.0f) ? (alpha * in4.z) : in4.z;
        out4.w = (in4.w < 0.0f) ? (alpha * in4.w) : in4.w;
        // Store result
        output4[vecIdx] = out4;
    }

    // Handle the tail (remaining elements)
    int tail_start = n_vec * 4;
    for (int i = tail_start + idx; i < N; i += stride) {
        float in_val = input[i];
        output[i] = (in_val < 0.0f) ? (alpha * in_val) : in_val;
    }
}

// External C API wrapper
extern "C" void leaky_RELU_optimized(
    float* output,
    const float* input,
    const int N,
    float alpha /*= 0.01f*/ = 0.01f) {
    // Launch parameters
    const int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    // Cap grid size by device maximum
    int max_grid_x;
    cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, 0);
    grid_size = (grid_size > max_grid_x ? max_grid_x : grid_size);

    dim3 block(block_size);
    dim3 grid(grid_size);

    // Kernel launch
    leaky_RELU_kernel_optimized<<<grid, block>>>(output, input, N, alpha);
    cudaDeviceSynchronize();
}