#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Configuration struct for launch parameters
struct Stencil2DBF16OptimizedConfig {
    dim3 block;
    dim3 grid;
};

// Shared-memory tiled 2D stencil kernel with fused boundary copy and vectorized loads (optimized)
__global__ void stencil_2d_bf16_kernel_optimized(
    __nv_bfloat16* u_new,
    const __nv_bfloat16* u_old,
    float r,
    int nx,
    int ny)
{
    extern __shared__ __nv_bfloat16 s_u[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global indices
    int j = bx * blockDim.x + tx;
    int i = by * blockDim.y + ty;

    // Dimensions of shared tile including halo
    int s_width = blockDim.x + 2;
    int s_height = blockDim.y + 2;
    int tile_elems = s_width * s_height;
    int tid = ty * blockDim.x + tx;
    int n_threads = blockDim.x * blockDim.y;

    // Phase 1: cooperative load into shared memory (with halo), vectorized bf16 loads
    const uint32_t* u_old_u32 = reinterpret_cast<const uint32_t*>(u_old);
    int pairs = tile_elems / 2;
    for (int pid = tid; pid < pairs; pid += n_threads) {
        int idx = pid * 2;            // shared-memory flat index for first of two
        int si  = idx / s_width;
        int sj0 = idx % s_width;
        int sj1 = sj0 + 1;
        int gi  = by * blockDim.y + (si - 1);
        int gj0 = bx * blockDim.x + (sj0 - 1);
        int gj1 = bx * blockDim.x + (sj1 - 1);
        int base_id = gi * ny + gj0;
        // both elements in-range for vector load
        if (gi >= 0 && gi < nx && gj0 >= 0 && gj1 < ny) {
            int vec_idx = base_id >> 1; // index of packed bf16 pair
            uint32_t packed = u_old_u32[vec_idx];
            union {
                uint32_t u32;
                __nv_bfloat16 v16[2];
            } tmp;
            tmp.u32 = packed;
            int sel0 = base_id & 1;
            s_u[idx]     = tmp.v16[sel0];
            s_u[idx + 1] = tmp.v16[sel0 ^ 1];
        } else {
            // fallback to scalar loads
            __nv_bfloat16 v0 = (gi >= 0 && gi < nx && gj0 >= 0 && gj0 < ny)
                               ? u_old[base_id]
                               : __float2bfloat16(0.0f);
            __nv_bfloat16 v1 = (gi >= 0 && gi < nx && gj1 >= 0 && gj1 < ny)
                               ? u_old[base_id + 1]
                               : __float2bfloat16(0.0f);
            s_u[idx]     = v0;
            s_u[idx + 1] = v1;
        }
    }
    __syncthreads();

    // Phase 2: compute stencil or copy boundary
    if (i < nx && j < ny) {
        float result;
        int gid = i * ny + j;
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            // boundary condition: direct copy
            result = __bfloat162float(u_old[gid]);
        } else {
            int si = ty + 1;
            int sj = tx + 1;
            int lidx = si * s_width + sj;
            float center = __bfloat162float(s_u[lidx]);
            float left   = __bfloat162float(s_u[lidx - 1]);
            float right  = __bfloat162float(s_u[lidx + 1]);
            float up     = __bfloat162float(s_u[lidx + s_width]);
            float down   = __bfloat162float(s_u[lidx - s_width]);
            result = center + r * (left + right + up + down - 4.0f * center);
        }
        u_new[gid] = __float2bfloat16(result);
    }
}

// External C wrapper using the optimized kernel
extern "C" void stencil_2d_bf16_optimized(
    __nv_bfloat16* u_new,
    const __nv_bfloat16* u_old,
    float r,
    int nx,
    int ny)
{
    // Launch parameters: blockDim.x must be multiple of 32 for coalescing
    Stencil2DBF16OptimizedConfig cfg;
    cfg.block = dim3(32, 8);
    cfg.grid  = dim3((ny + cfg.block.x - 1) / cfg.block.x,
                     (nx + cfg.block.y - 1) / cfg.block.y);
    // Shared memory size: (blockDim.x+2)*(blockDim.y+2) elements
    size_t shared_bytes = (cfg.block.x + 2) * (cfg.block.y + 2) * sizeof(__nv_bfloat16);

    stencil_2d_bf16_kernel_optimized<<<cfg.grid, cfg.block, shared_bytes>>>(
        u_new, u_old, r, nx, ny);
    cudaDeviceSynchronize();
}
