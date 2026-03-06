#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <utility>

// Polynomial-based Silu(x) * g kernel replaced with exact expf-based Silu
extern "C" __global__ void silu_and_mul_kernel_optimized(
    const float* __restrict__ in,    // shape B x (2*D)
    float*       __restrict__ out,   // shape B x D
    int32_t      B,
    int32_t      D)
{
    // D is guaranteed a multiple of 4
    int32_t D4   = D >> 2;            // number of float4 elements
    int32_t tile = blockIdx.x;        // tile index along feature dim
    int32_t b    = blockIdx.y;        // batch index
    int32_t tid  = threadIdx.x;
    int32_t idx  = tile * blockDim.x + tid;
    if (idx >= D4) return;

    // load pointers: each batch has 2*D floats == 2*D4 float4’s
    const float4* batch_in4 = reinterpret_cast<const float4*>(in) + size_t(b) * (2 * D4);
    const float4* x4_ptr    = batch_in4 + idx;        // x begins at offset 0
    const float4* g4_ptr    = batch_in4 + D4 + idx;   // g begins at offset D4

    float4 x4 = *x4_ptr;
    float4 g4 = *g4_ptr;

    // compute silu(x) * g exactly using expf
    float4 o4;

    // lane x
    {
        float xv = x4.x;
        float s  = 1.0f + expf(-xv);
        o4.x     = (xv / s) * g4.x;
    }
    // lane y
    {
        float xv = x4.y;
        float s  = 1.0f + expf(-xv);
        o4.y     = (xv / s) * g4.y;
    }
    // lane z
    {
        float xv = x4.z;
        float s  = 1.0f + expf(-xv);
        o4.z     = (xv / s) * g4.z;
    }
    // lane w
    {
        float xv = x4.w;
        float s  = 1.0f + expf(-xv);
        o4.w     = (xv / s) * g4.w;
    }

    // write back
    float4* batch_out4 = reinterpret_cast<float4*>(out) + size_t(b) * D4;
    batch_out4[idx]    = o4;
}

extern "C" void silu_and_mul_optimized(
    const float* in, float* out,
    int32_t B, int32_t D) {
    int32_t D4    = D >> 2;
    int32_t tiles = (D4 + 256 - 1) / 256;
    dim3 block(256, 1, 1);
    dim3 grid(tiles, B, 1);
    
    silu_and_mul_kernel_optimized<<<grid, block>>>(in, out, B, D);
    cudaDeviceSynchronize();
}