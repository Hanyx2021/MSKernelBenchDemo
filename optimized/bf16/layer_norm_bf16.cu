#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

// Global device scalars for cooperative reduction and normalization
__device__ float d_sum_coop;
__device__ float d_sqsum_coop;
__device__ float d_mean_coop;
__device__ float d_var_coop;

// Fused cooperative-grid kernel for mean/variance and layer normalization
__global__ void fused_layer_norm_bf16_kernel_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float epsilon,
    int N) {
    // Obtain the grid group for full-grid synchronization
    grid_group grid = this_grid();

    // Initialize global accumulators once
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_sum_coop = 0.0f;
        d_sqsum_coop = 0.0f;
    }
    grid.sync();

    // Compute local sums via grid-stride loop
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum   = 0.0f;
    float local_sqsum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        float val = __bfloat162float(input[i]);
        local_sum   += val;
        local_sqsum += val * val;
    }

    // Warp-level shuffle reduction
    unsigned int mask = 0xffffffffu;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        float tmp_sum   = __shfl_down_sync(mask, local_sum, offset);
        float tmp_sqsum = __shfl_down_sync(mask, local_sqsum, offset);
        local_sum   += tmp_sum;
        local_sqsum += tmp_sqsum;
    }

    // Atomic update of global accumulators by first lane of each warp
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&d_sum_coop, local_sum);
        atomicAdd(&d_sqsum_coop, local_sqsum);
    }
    grid.sync();

    // Single thread computes mean and variance and broadcasts to device globals
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum   = d_sum_coop;
        float sqsum = d_sqsum_coop;
        float mean  = sum / N;
        float var   = sqsum / N - mean * mean;
        d_mean_coop = mean;
        d_var_coop  = var;
    }
    grid.sync();

    // Final normalization pass (grid-stride) and write output
    float mean    = d_mean_coop;
    float var     = d_var_coop;
    float inv_std = rsqrtf(var + epsilon);
    for (int i = idx; i < N; i += stride) {
        float x = __bfloat162float(input[i]);
        float w = __bfloat162float(weight[i]);
        float b = __bfloat162float(bias[i]);
        float normalized = (x - mean) * inv_std;
        float result     = w * normalized + b;
        output[i]        = __float2bfloat16(result);
    }
}

// External C wrapper for launching the cooperative kernel
extern "C" void layer_norm_bf16_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const float epsilon,
    const int N) {
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    blocks = blocks > 1024 ? 1024 : blocks;
    void* args[] = { &output, &input, &weight, &bias, (void*)&epsilon, (void*)&N };

    // Launch as a cooperative-grid kernel
    cudaLaunchCooperativeKernel(
        (void*)fused_layer_norm_bf16_kernel_optimized,
        dim3(blocks), dim3(threadsPerBlock),
        args);
    cudaDeviceSynchronize();
}