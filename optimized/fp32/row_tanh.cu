#include <cuda_runtime.h>
#include <math.h>

// Configuration struct for optimized launch parameters
struct RowTanhOptimizedConfig {
    int block_size;
    int grid_size;
};

// Optimized kernel: vectorized 128-bit loads/stores via float4 and tail handling
__global__ void row_tanh_kernel_optimized(
    float* out,
    const float* input,
    int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N4 = N >> 2;        // number of float4 elements
    int R  = N & 3;         // tail elements (0..3)
    int total = N4 + R;     // total "work" units (vector + tail)

    if (tid < N4) {
        // Vectorized processing
        const float4* in4 = reinterpret_cast<const float4*>(input);
        float4 v = in4[tid];
        float4 r;
        r.x = tanhf(v.x);
        r.y = tanhf(v.y);
        r.z = tanhf(v.z);
        r.w = tanhf(v.w);
        float4* out4 = reinterpret_cast<float4*>(out);
        out4[tid] = r;
    } else if (tid < total) {
        // Tail processing for remaining 1-3 elements
        int base = N4 << 2;               // start index of tail in elements
        int offset = tid - N4;            // 0..R-1
        int idx = base + offset;
        out[idx] = tanhf(input[idx]);
    }
}

// External C wrapper with optimized suffix
extern "C" void row_tanh_optimized(
    float* out,
    const float* input,
    const int N) {
    RowTanhOptimizedConfig config;
    config.block_size = 256;
    int N4 = N >> 2;
    int R  = N & 3;
    int total_threads = N4 + R;
    config.grid_size = (total_threads + config.block_size - 1) / config.block_size;

    dim3 block(config.block_size, 1, 1);
    dim3 grid(config.grid_size, 1, 1);
    row_tanh_kernel_optimized<<<grid, block>>>(out, input, N);
    cudaDeviceSynchronize();
}