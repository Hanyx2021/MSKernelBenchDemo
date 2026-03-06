#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused RMS reduction + normalization for small N with single load of X
__global__ void rms_fused_kernel_optimized(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    float epsilon,
    int N) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned)N) return;

    // Load X once and cache in register
    float x = input[tid];
    float sum = x * x;

    // Warp-level reduction
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Shared memory for warp sums and final RMS
    __shared__ float warp_sums[32];
    __shared__ float s_rms;

    unsigned lane = tid & (warpSize - 1);
    unsigned wid  = tid / warpSize;
    if (lane == 0) {
        warp_sums[wid] = sum;
    }
    __syncthreads();

    // First warp reduces the warp-level sums to a block sum and compute RMS
    if (wid == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = (lane < (unsigned)num_warps ? warp_sums[lane] : 0.0f);
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (lane == 0) {
            s_rms = sqrtf(block_sum / N + epsilon);
        }
    }
    __syncthreads();

    // Normalize using cached x and write output
    float rms = s_rms;
    float x_hat = x / rms;
    output[tid] = weight[tid] * x_hat + bias[tid];
}

// Block-level reduction kernel for large N (unchanged)
__global__ void rms_kernel_optimized(
    const float* input,
    float* rms,
    int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float v = input[i];
        sum += v * v;
    }
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    unsigned lane = threadIdx.x & (warpSize - 1);
    unsigned wid  = threadIdx.x / warpSize;
    __shared__ float warp_sums[32];
    if (lane == 0) {
        warp_sums[wid] = sum;
    }
    __syncthreads();
    if (wid == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float block_sum = (lane < num_warps ? warp_sums[lane] : 0.0f);
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(rms, block_sum);
        }
    }
}

// Normalization kernel for large N (unchanged)
__global__ void rms_norm_kernel_optimized(
    float* output,
    const float* input,
    const float* d_rms,
    const float* weight,
    const float* bias,
    float epsilon,
    int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float s = *d_rms;
    float rms = sqrtf(s / N + epsilon);
    for (int i = tid; i < N; i += stride) {
        float x = input[i];
        float x_hat = x / rms;
        output[i] = weight[i] * x_hat + bias[i];
    }
}

// External C wrapper calling optimized kernels
extern "C" void rms_norm_optimized(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float epsilon,
    const int N) {
    const int kMaxThreads = 1024;
    if (N <= kMaxThreads) {
        // Single-block fused kernel for small N
        int threads = N;
        rms_fused_kernel_optimized<<<1, threads>>>(
            output, input, weight, bias, epsilon, N);
        cudaDeviceSynchronize();
    } else {
        // Fallback for large N: two-kernel approach
        float* d_rms = nullptr;
        cudaMalloc(&d_rms, sizeof(float));
        cudaMemset(d_rms, 0, sizeof(float));

        const int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        if (blocks > 1024) blocks = 1024;

        rms_kernel_optimized<<<blocks, threadsPerBlock>>>(input, d_rms, N);
        cudaDeviceSynchronize();

        rms_norm_kernel_optimized<<<blocks, threadsPerBlock>>>(
            output, input, d_rms, weight, bias, epsilon, N);
        cudaDeviceSynchronize();

        cudaFree(d_rms);
    }
}