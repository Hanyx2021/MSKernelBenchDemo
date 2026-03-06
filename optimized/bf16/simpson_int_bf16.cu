#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <cstdint> // for uint32_t

// Fixed optimized Simpson integration: scalar bf16 loads & conversions
__global__ void simpson_int_bf16_kernel_optimized(
    const __nv_bfloat16* y_samples,
    float* partial_sum,
    float  a,
    float  b,
    int    N
) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;
    int S      = (N - 1) / 2;      // number of Simpson segments
    float h    = (b - a) / (N - 1);
    float thread_sum = 0.0f;

    // Each pos is one Simpson segment [i, i+1, i+2] with i even
    for (int pos = tid; pos < S; pos += stride) {
        int i = pos * 2;

        // Scalar bf16 → float conversions
        float f0 = __bfloat162float(y_samples[i]);
        float f1 = __bfloat162float(y_samples[i + 1]);
        float f2 = __bfloat162float(y_samples[i + 2]);

        thread_sum += (h / 3.0f) * (f0 + 4.0f * f1 + f2);
    }

    // Block‐wide reduction with CUB
    typedef cub::BlockReduce<float, 256> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    float block_sum = BlockReduceT(temp_storage).Sum(thread_sum);

    if (threadIdx.x == 0) {
        atomicAdd(partial_sum, block_sum);
    }
}

// Finalize and convert accumulated sum to bfloat16
__global__ void finalize_simpson_integral_kernel_optimized(
    __nv_bfloat16* result,
    float* integral_sum,
    float a,
    float b,
    int N
) {
    float integral_value = *integral_sum;
    *result = __float2bfloat16(integral_value);
}

extern "C" void simpson_int_bf16_optimized(
    const __nv_bfloat16* y_samples,
    __nv_bfloat16* result,
    float a,
    float b,
    int N
) {
    // Allocate and zero the accumulator
    float* d_integral_sum = nullptr;
    cudaMalloc(&d_integral_sum, sizeof(float));
    cudaMemset(d_integral_sum, 0, sizeof(float));

    int S = (N - 1) / 2;
    const int threadsPerBlock = 256;
    int blocks = (S + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;

    // Compute partial sums with block-wise reduction
    simpson_int_bf16_kernel_optimized<<<blocks, threadsPerBlock>>>(
        y_samples, d_integral_sum, a, b, N
    );

    // Finalize the integral and store result
    finalize_simpson_integral_kernel_optimized<<<1, 1>>>(
        result, d_integral_sum, a, b, N
    );

    cudaDeviceSynchronize();
    cudaFree(d_integral_sum);
}