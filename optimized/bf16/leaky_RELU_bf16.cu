#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int VEC = 2;

// Updated optimized kernel: vectorized 2-way bfloat16 operations with wide loads/stores
__global__ void leaky_RELU_bf16_kernel_optimized(
    __nv_bfloat16*       output_bf,
    const __nv_bfloat16* input_bf,
    uint32_t*            output_vec,
    const uint32_t*      input_vec,
    int                   Nvec,
    int                   tail,
    float                 alpha) {
    int base   = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // Process full 2-element vectors
    for (int i = base; i < Nvec; i += stride) {
        uint32_t packed = __ldg(input_vec + i);
        uint16_t raw0 = static_cast<uint16_t>(packed & 0xFFFFu);
        uint16_t raw1 = static_cast<uint16_t>(packed >> 16);
        __nv_bfloat16 lane0 = *reinterpret_cast<__nv_bfloat16*>(&raw0);
        __nv_bfloat16 lane1 = *reinterpret_cast<__nv_bfloat16*>(&raw1);
        float f0 = __bfloat162float(lane0);
        float f1 = __bfloat162float(lane1);
        float o0 = fmaxf(f0, 0.0f) + alpha * fminf(f0, 0.0f);
        float o1 = fmaxf(f1, 0.0f) + alpha * fminf(f1, 0.0f);
        __nv_bfloat16 bf0 = __float2bfloat16(o0);
        __nv_bfloat16 bf1 = __float2bfloat16(o1);
        uint16_t out0 = *reinterpret_cast<uint16_t*>(&bf0);
        uint16_t out1 = *reinterpret_cast<uint16_t*>(&bf1);
        uint32_t out_packed = (static_cast<uint32_t>(out1) << 16) | static_cast<uint32_t>(out0);
        output_vec[i] = out_packed;
    }
    // Handle tail elements (N mod VEC)
    int tid = base;
    if (tid < tail) {
        int idx = Nvec * VEC + tid;
        __nv_bfloat16 in_val = __ldg(input_bf + idx);
        float fv = __bfloat162float(in_val);
        float ov = fmaxf(fv, 0.0f) + alpha * fminf(fv, 0.0f);
        output_bf[idx] = __float2bfloat16(ov);
    }
}

extern "C" void leaky_RELU_bf16_optimized(
    __nv_bfloat16*       output,
    const __nv_bfloat16* input,
    const int            N,
    float                 alpha = 0.01f) {
    // Compute vector and tail counts
    const int V = VEC;
    int Nvec = N / V;
    int tail = N - Nvec * V;

    // Cast pointers for wide operations
    uint32_t*       output_vec = reinterpret_cast<uint32_t*>(output);
    const uint32_t* input_vec  = reinterpret_cast<const uint32_t*>(input);

    // Launch configuration
    const int blockSize = 256;
    int gridSize = (Nvec + blockSize - 1) / blockSize;
    if (gridSize < 1) gridSize = 1;
    dim3 block(blockSize, 1, 1);
    dim3 grid(gridSize, 1, 1);

    // Launch optimized kernel
    leaky_RELU_bf16_kernel_optimized<<<grid, block>>> (
        output,
        input,
        output_vec,
        input_vec,
        Nvec,
        tail,
        alpha);
    cudaDeviceSynchronize();
}