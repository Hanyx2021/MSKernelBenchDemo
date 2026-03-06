#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Tile size for splitting the D dimension across blocks
static constexpr int TILE = 128;

// Optimized kernel: each block processes a TILE-sized tile of the D dimension for one batch index
__global__ void silu_and_mul_bf16_kernel_optimized(
    const __nv_bfloat16* in, __nv_bfloat16* out,
    int32_t B, int32_t D)
{
    int b = blockIdx.y;
    if (b >= B) return;

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    // Pointers to the x and g subtensors for batch b
    const __nv_bfloat16* row = in + b * 2 * D;
    const __nv_bfloat16* x_ptr = row;
    const __nv_bfloat16* g_ptr = row + D;
    __nv_bfloat16* o = out + b * D;

    float xv = __bfloat162float(x_ptr[d]);
    float gv = __bfloat162float(g_ptr[d]);
    float silu_xv = xv / (1.0f + expf(-xv));
    o[d] = __float2bfloat16(silu_xv * gv);
}

extern "C" void silu_and_mul_bf16_optimized(
    const __nv_bfloat16* in, __nv_bfloat16* out,
    int32_t B, int32_t D)
{
    dim3 block(TILE, 1, 1);
    int grid_x = (D + TILE - 1) / TILE;
    dim3 grid(grid_x, B, 1);

    silu_and_mul_bf16_kernel_optimized<<<grid, block>>>(in, out, B, D);
    cudaDeviceSynchronize();
}
