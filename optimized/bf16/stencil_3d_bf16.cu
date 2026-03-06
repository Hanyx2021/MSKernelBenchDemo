#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Block dimensions for tiling
#define BX 8
#define BY 8
#define BZ 8

// Fused stencil and boundary-copy kernel
__global__ void stencil_3d_bf16_kernel_optimized(
    __nv_bfloat16* u_new,
    const __nv_bfloat16* u_old,
    float r,
    int nx, int ny, int nz) {
    // Shared tile with 1-cell halo on each face
    __shared__ __nv_bfloat16 tile[BX+2][BY+2][BZ+2];

    // Thread indices within block
    int tx = threadIdx.x; // k dimension
    int ty = threadIdx.y; // j dimension
    int tz = threadIdx.z; // i dimension
    int threadLinear = tz * (BX * BY) + ty * BX + tx;

    const int threadsPerBlock = BX * BY * BZ;
    const int tileSize = (BX + 2) * (BY + 2) * (BZ + 2);

    // Cooperative load of tile (including halo)
    for (int index = threadLinear; index < tileSize; index += threadsPerBlock) {
        int si = index / ((BY + 2) * (BZ + 2));
        int rem = index % ((BY + 2) * (BZ + 2));
        int sj = rem / (BZ + 2);
        int sk = rem % (BZ + 2);

        // Map shared coords back to global coords (with halo offset)
        int gi = blockIdx.z * BX + (si - 1);
        int gj = blockIdx.y * BY + (sj - 1);
        int gk = blockIdx.x * BZ + (sk - 1);

        __nv_bfloat16 val;
        if (gi >= 0 && gi < nx && gj >= 0 && gj < ny && gk >= 0 && gk < nz) {
            int gidx = gi * (ny * nz) + gj * nz + gk;
            val = u_old[gidx];
        } else {
            // Out-of-bounds halo
            val = __float2bfloat16(0.0f);
        }
        tile[si][sj][sk] = val;
    }

    // Ensure full tile is loaded
    __syncthreads();

    // Compute this thread's global indices
    int gi = blockIdx.z * BX + tz;
    int gj = blockIdx.y * BY + ty;
    int gk = blockIdx.x * BZ + tx;

    // Only threads covering valid global points
    if (gi < nx && gj < ny && gk < nz) {
        // Shared-memory center indices
        int si = tz + 1;
        int sj = ty + 1;
        int sk = tx + 1;
        int outIdx = gi * (ny * nz) + gj * nz + gk;

        // Interior stencil or boundary copy
        if (gi >= 1 && gi < nx - 1 &&
            gj >= 1 && gj < ny - 1 &&
            gk >= 1 && gk < nz - 1) {
            float center = __bfloat162float(tile[si][sj][sk]);
            float left   = __bfloat162float(tile[si-1][sj][sk]);
            float right  = __bfloat162float(tile[si+1][sj][sk]);
            float up     = __bfloat162float(tile[si][sj+1][sk]);
            float down   = __bfloat162float(tile[si][sj-1][sk]);
            float front  = __bfloat162float(tile[si][sj][sk+1]);
            float back   = __bfloat162float(tile[si][sj][sk-1]);

            float result = center + r * (left + right + up + down + front + back - 6.0f * center);
            u_new[outIdx] = __float2bfloat16(result);
        } else {
            // Boundary copy from u_old via shared tile
            u_new[outIdx] = tile[si][sj][sk];
        }
    }
}

// External C wrapper for fused kernel
extern "C" void stencil_3d_bf16_optimized(
    __nv_bfloat16* u_new,
    const __nv_bfloat16* u_old,
    float r,
    int nx, int ny, int nz) {
    // Launch with tiled block dims mapping to (k,j,i)
    dim3 block(BZ, BY, BX);
    dim3 grid(
        (nz + BZ - 1) / BZ,
        (ny + BY - 1) / BY,
        (nx + BX - 1) / BX
    );

    stencil_3d_bf16_kernel_optimized<<<grid, block>>>(u_new, u_old, r, nx, ny, nz);
    cudaDeviceSynchronize();
}