#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cmath>

// Maximum D/2 supported in constant memory (adjust if needed)
#define MAX_D2 8192

// Precomputed theta values in constant memory
__constant__ float theta_const[MAX_D2];

// Configuration structure for optimized launch parameters
struct RopeBf16OptimizedConfig {
    dim3 grid;
    dim3 block;
};

// Optimized kernel with fused sincos and 32-bit vectorized loads/stores
__global__ void rope_bf16_kernel_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int M,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d2 = D / 2;
    int total_pairs = M * d2;
    if (idx >= total_pairs) return;

    int m = idx / d2;
    int pair = idx % d2;

    // Load two bfloat16s at once via 32-bit transaction
    const __nv_bfloat162* input32 = reinterpret_cast<const __nv_bfloat162*>(input);
    __nv_bfloat162 packed_in = __ldg(&input32[idx]);

    // De-pack manually using scalar conversion
    float q0 = __bfloat162float(packed_in.x);
    float q1 = __bfloat162float(packed_in.y);

    // Lookup precomputed theta via read-only cache
    float theta = __ldg(&theta_const[pair]);
    float m_theta = static_cast<float>(m) * theta;

    // Compute sin and cos via fused intrinsic
    float sinv, cosv;
    sincosf(m_theta, &sinv, &cosv);

    // Apply rotary transform
    float2 out;
    out.x = q0 * cosv - q1 * sinv;
    out.y = q0 * sinv + q1 * cosv;

    // Pack and store results in one 32-bit transaction
    __nv_bfloat162 packed_out;
    packed_out.x = __float2bfloat16(out.x);
    packed_out.y = __float2bfloat16(out.y);

    __nv_bfloat162* output32 = reinterpret_cast<__nv_bfloat162*>(output);
    output32[idx] = packed_out;
}

// External C function wrapper
extern "C" void rope_bf16_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int M,
    int D,
    float base
) {
    int d2 = D / 2;
    // Precompute theta on host
    std::vector<float> theta_host(d2);
    for (int k = 0; k < d2; ++k) {
        theta_host[k] = powf(base, -2.0f * static_cast<float>(k) / static_cast<float>(D));
    }
    // Copy to constant memory
    cudaMemcpyToSymbol(theta_const, theta_host.data(), d2 * sizeof(float));

    // Launch kernel with optimized vectorized load/store
    int total_pairs = M * d2;
    const int threads = 512;
    int blocks = (total_pairs + threads - 1) / threads;

    RopeBf16OptimizedConfig config;
    config.block = dim3(threads, 1, 1);
    config.grid  = dim3(blocks,  1, 1);

    rope_bf16_kernel_optimized<<<config.grid, config.block>>>(output, input, M, D);
    cudaDeviceSynchronize();
}
