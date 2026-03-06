#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Vectorized optimized sigmoid kernel for bf16 data (fixed bit-casting)
__global__ void sigmoid_bf16_kernel_optimized(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    int N) {
    // Treat the bf16 array as an array of packed 32-bit words
    const uint32_t* __restrict__ in32  = reinterpret_cast<const uint32_t*>(input);
    uint32_t*       __restrict__ out32 = reinterpret_cast<uint32_t*>(out);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vi = static_cast<size_t>(idx) * 2;
    if (vi >= static_cast<size_t>(N)) return;

    if (vi + 1 < static_cast<size_t>(N)) {
        // Load two bf16 elements packed in one 32-bit word via read-only cache
        uint32_t raw = __ldg(in32 + idx);

        uint16_t raw0 = static_cast<uint16_t>(raw & 0xFFFF);
        uint16_t raw1 = static_cast<uint16_t>(raw >> 16);

        // Bit-cast those 16 bits into __nv_bfloat16
        __nv_bfloat16 h0 = *reinterpret_cast<const __nv_bfloat16*>(&raw0);
        __nv_bfloat16 h1 = *reinterpret_cast<const __nv_bfloat16*>(&raw1);

        float f0 = __bfloat162float(h0);
        float f1 = __bfloat162float(h1);

        // Sigmoid computation
        float s0 = 1.0f / (1.0f + expf(-f0));
        float s1 = 1.0f / (1.0f + expf(-f1));

        // Convert back to bf16
        __nv_bfloat16 r0 = __float2bfloat16(s0);
        __nv_bfloat16 r1 = __float2bfloat16(s1);

        // Extract the raw 16-bit fields and repack into a uint32_t
        uint16_t rv0 = *reinterpret_cast<const uint16_t*>(&r0);
        uint16_t rv1 = *reinterpret_cast<const uint16_t*>(&r1);
        uint32_t out_raw = (static_cast<uint32_t>(rv1) << 16) | static_cast<uint32_t>(rv0);

        out32[idx] = out_raw;
    } else {
        // Handle the last odd element
        __nv_bfloat16 h0 = __ldg(input + vi);
        float f0 = __bfloat162float(h0);
        float s0 = 1.0f / (1.0f + expf(-f0));
        __nv_bfloat16 r0 = __float2bfloat16(s0);
        out[vi] = r0;
    }
}

// External C wrapper with optimized launch configuration
extern "C" void sigmoid_bf16_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int N) {
    const int blockSize = 256;
    int N2 = (N + 1) / 2;
    int gridSize = (N2 + blockSize - 1) / blockSize;

    dim3 block(blockSize, 1, 1);
    dim3 grid(gridSize, 1, 1);

    sigmoid_bf16_kernel_optimized<<<grid, block>>>(out, input, N);
    cudaDeviceSynchronize();
}
