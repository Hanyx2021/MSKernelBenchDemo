#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Combined mean and squared-sum reduction kernel
__global__ void mean_var_kernel_optimized(
    const float* __restrict__ input,
    float* mean_sum,
    float* sqsum_sum,
    int N) {
    extern __shared__ float s_data[];
    // First half for sums, second half for squared sums
    float* s_sum = s_data;
    float* s_sqsum = s_data + (blockDim.x / warpSize);

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;

    // Grid-strided loop to accumulate sum and sum of squares
    for (int i = tid; i < N; i += stride) {
        float x = __ldg(&input[i]);
        local_sum += x;
        local_sqsum += x * x;
    }

    unsigned int fullMask = 0xffffffff;
    // In-warp reduction for both sums
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum   += __shfl_down_sync(fullMask, local_sum, offset);
        local_sqsum += __shfl_down_sync(fullMask, local_sqsum, offset);
    }

    unsigned int lane = threadIdx.x & (warpSize - 1);
    unsigned int warpId = threadIdx.x / warpSize;

    // Write per-warp results to shared memory
    if (lane == 0) {
        s_sum[warpId]   = local_sum;
        s_sqsum[warpId] = local_sqsum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warpId == 0) {
        float block_sum   = (lane < (blockDim.x / warpSize)) ? s_sum[lane]   : 0.0f;
        float block_sqsum = (lane < (blockDim.x / warpSize)) ? s_sqsum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            block_sum   += __shfl_down_sync(fullMask, block_sum, offset);
            block_sqsum += __shfl_down_sync(fullMask, block_sqsum, offset);
        }
        if (lane == 0) {
            atomicAdd(mean_sum,   block_sum);
            atomicAdd(sqsum_sum,  block_sqsum);
        }
    }
}

// Layer normalization kernel
__global__ void layer_norm_kernel_optimized(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean_val,
    const float* __restrict__ var_val,
    float epsilon,
    int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float mean = __ldg(mean_val);
    float var  = __ldg(var_val);
    float inv_std = rsqrtf(var + epsilon);

    for (int i = tid; i < N; i += stride) {
        float in = __ldg(&input[i]);
        float w  = __ldg(&weight[i]);
        float b  = __ldg(&bias[i]);
        float normalized = (in - mean) * inv_std;
        output[i] = w * normalized + b;
    }
}

// External C wrapper
extern "C" void layer_norm_optimized(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float epsilon,
    const int N) {
    // Allocate device temporaries
    float *d_sum, *d_sqsum, *d_mean_val, *d_var_val;
    cudaMalloc(&d_sum,      sizeof(float));
    cudaMalloc(&d_sqsum,    sizeof(float));
    cudaMalloc(&d_mean_val, sizeof(float));
    cudaMalloc(&d_var_val,  sizeof(float));
    cudaMemset(d_sum,   0, sizeof(float));
    cudaMemset(d_sqsum, 0, sizeof(float));

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;
    const int hostWarpSize = 32;
    int warpCount = threadsPerBlock / hostWarpSize;
    size_t shared_mem = 2 * warpCount * sizeof(float);

    // Cache preference
    cudaFuncSetCacheConfig(mean_var_kernel_optimized,   cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(layer_norm_kernel_optimized, cudaFuncCachePreferL1);

    // 1) Compute sum and sum of squares in one pass
    mean_var_kernel_optimized<<<blocks, threadsPerBlock, shared_mem>>>(
        input, d_sum, d_sqsum, N);
    cudaDeviceSynchronize();

    // Retrieve results to host
    float h_sum, h_sqsum;
    cudaMemcpy(&h_sum,   d_sum,   sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sqsum, d_sqsum, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute mean and variance
    float h_mean = h_sum / N;
    float h_var  = h_sqsum / N - h_mean * h_mean;
    cudaMemcpy(d_mean_val, &h_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_val,  &h_var,  sizeof(float), cudaMemcpyHostToDevice);

    // 2) Layer normalization
    layer_norm_kernel_optimized<<<blocks, threadsPerBlock>>>(
        output, input, weight, bias,
        d_mean_val, d_var_val,
        epsilon, N);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(d_sum);
    cudaFree(d_sqsum);
    cudaFree(d_mean_val);
    cudaFree(d_var_val);
}