#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused batch normalization kernel: computes mean/variance and applies normalization in one pass
// Each block handles one feature (channel) j
// blockDim.x = min(N, 1024)
// gridDim.x = C

__global__ void batch_norm_kernel_optimized(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    const float epsilon,
    const int N,
    const int C) {
    // feature index
    int j = blockIdx.x;
    // thread index within block
    int tid = threadIdx.x;
    int T = blockDim.x; // number of threads used

    // Shared memory for storing values and partial reductions
    __shared__ float s_x[1024];
    __shared__ float s_sum[1024];
    __shared__ float s_sumsq[1024];

    // Load input values for this feature into shared memory and compute partial sums
    float x = 0.0f;
    if (tid < N) {
        int idx = tid * C + j;
        x = input[idx];
    }
    s_x[tid] = x;
    s_sum[tid] = x;
    s_sumsq[tid] = x * x;
    __syncthreads();

    // Parallel reduction for sum and sum of squares
    for (int stride = (T + 1) / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < T) {
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 computes mean and inverse stddev
    float mean = 0.0f;
    float invstd = 0.0f;
    if (tid == 0) {
        float sum = s_sum[0];
        float sqsum = s_sumsq[0];
        mean = sum / float(N);
        float var = sqsum / float(N) - mean * mean;
        invstd = rsqrtf(var + epsilon);
        // Store mean and invstd in shared slots for broadcast
        s_sum[0] = mean;
        s_sumsq[0] = invstd;
    }
    __syncthreads();

    // Broadcast mean and invstd
    mean = s_sum[0];
    invstd = s_sumsq[0];
    // Load scale and shift parameters for this feature
    float scale = gamma[j];
    float shift = beta[j];

    // Apply normalization and write output
    if (tid < N) {
        float xn = (s_x[tid] - mean) * invstd;
        output[tid * C + j] = xn * scale + shift;
    }
}

extern "C" void batch_norm_optimized(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    const float epsilon,
    const int N,
    const int C) {
    // Launch one block per feature, threads per block = min(N, 1024)
    int threads = (N < 1024) ? N : 1024;
    int blocks = C;
    batch_norm_kernel_optimized<<<blocks, threads>>>(
        output, input, gamma, beta, epsilon, N, C);
    cudaDeviceSynchronize();
}
