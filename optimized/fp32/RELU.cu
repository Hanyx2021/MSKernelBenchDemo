#include <cuda.h>
#include <cuda_runtime.h>

// Configuration struct for optimized launch parameters
typedef struct {
    int blockSize;
    int gridSize;
} RELUConfigOptimized;

// Optimized RELU kernel: vectorized float4 loads/stores and tail handling
__global__ void RELU_kernel_optimized(
    float* output,
    const float* input,
    int N) {
    // Each thread processes one float4 (4 floats)
    int global_idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int num4 = N >> 2;  // number of full float4 elements (N/4)

    if (global_idx4 < num4) {
        // reinterpret pointers as float4 pointers
        const float4* in4 = reinterpret_cast<const float4*>(input);
        float4* out4 = reinterpret_cast<float4*>(output);
        
        // load, apply RELU, and store
        float4 v = in4[global_idx4];
        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
        out4[global_idx4] = v;
    }

    // Handle tail elements (when N is not a multiple of 4)
    // Only thread 0 handles the remainder to amortize overhead
    if (global_idx4 == 0) {
        int tail_start = (num4 << 2);  // tail_start = num4 * 4
        for (int idx = tail_start; idx < N; ++idx) {
            output[idx] = fmaxf(0.0f, input[idx]);
        }
    }
}

// External C wrapper with optimized launch configuration
extern "C" void RELU_optimized(
    float* output,
    const float* input,
    int N) {
    // Set up optimized block and grid sizes for float4 processing
    RELUConfigOptimized config;
    config.blockSize = 256;  // multiple of 32 for warp efficiency
    int num4 = (N + 3) >> 2;  // ceil(N/4)
    config.gridSize = (num4 + config.blockSize - 1) / config.blockSize;

    dim3 block(config.blockSize, 1, 1);
    dim3 grid(config.gridSize, 1, 1);

    // Launch optimized kernel
    RELU_kernel_optimized<<<grid, block>>>(output, input, N);
    cudaDeviceSynchronize();
}
