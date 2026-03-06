#include <cuda.h>
#include <cuda_runtime.h>

// Optimized 2D stencil kernel with 2D shared-memory tiling
extern "C" __global__ void stencil_2d_kernel_optimized(
    float* u_new,
    const float* u_old,
    float r,
    int nx,
    int ny) {
    // Global indices: threadIdx.x -> j (column), threadIdx.y -> i (row)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared tile dimensions (including 1-cell halo border)
    int tile_width  = blockDim.x + 2;
    int tile_height = blockDim.y + 2;
    int local_x     = threadIdx.x + 1;
    int local_y     = threadIdx.y + 1;

    // Dynamic shared memory
    extern __shared__ float tile[];

    // Load center element
    if (i < nx && j < ny) {
        tile[local_y * tile_width + local_x] = u_old[i * ny + j];
    } else {
        tile[local_y * tile_width + local_x] = 0.0f;
    }

    // Load halo elements cooperatively
    // Top halo
    if (threadIdx.y == 0) {
        int ii = i - 1;
        int jj = j;
        float v = (ii >= 0 && jj >= 0 && jj < ny) ? u_old[ii * ny + jj] : 0.0f;
        tile[0 * tile_width + local_x] = v;
    }
    // Bottom halo
    if (threadIdx.y == blockDim.y - 1) {
        int ii = i + 1;
        int jj = j;
        float v = (ii < nx && jj >= 0 && jj < ny) ? u_old[ii * ny + jj] : 0.0f;
        tile[(tile_height - 1) * tile_width + local_x] = v;
    }
    // Left halo
    if (threadIdx.x == 0) {
        int ii = i;
        int jj = j - 1;
        float v = (ii >= 0 && ii < nx && jj >= 0) ? u_old[ii * ny + jj] : 0.0f;
        tile[local_y * tile_width + 0] = v;
    }
    // Right halo
    if (threadIdx.x == blockDim.x - 1) {
        int ii = i;
        int jj = j + 1;
        float v = (ii >= 0 && ii < nx && jj < ny) ? u_old[ii * ny + jj] : 0.0f;
        tile[local_y * tile_width + (tile_width - 1)] = v;
    }
    // Corner halos
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int ii = i - 1;
        int jj = j - 1;
        float v = (ii >= 0 && jj >= 0) ? u_old[ii * ny + jj] : 0.0f;
        tile[0 * tile_width + 0] = v;
    }
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        int ii = i + 1;
        int jj = j - 1;
        float v = (ii < nx && jj >= 0) ? u_old[ii * ny + jj] : 0.0f;
        tile[(tile_height - 1) * tile_width + 0] = v;
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        int ii = i - 1;
        int jj = j + 1;
        float v = (ii >= 0 && jj < ny) ? u_old[ii * ny + jj] : 0.0f;
        tile[0 * tile_width + (tile_width - 1)] = v;
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        int ii = i + 1;
        int jj = j + 1;
        float v = (ii < nx && jj < ny) ? u_old[ii * ny + jj] : 0.0f;
        tile[(tile_height - 1) * tile_width + (tile_width - 1)] = v;
    }

    __syncthreads();

    // Write output
    if (i < nx && j < ny) {
        int idx = i * ny + j;
        // Boundary cells: copy directly
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            u_new[idx] = u_old[idx];
        } else {
            float center = tile[local_y * tile_width + local_x];
            float left   = tile[local_y * tile_width + (local_x - 1)];
            float right  = tile[local_y * tile_width + (local_x + 1)];
            float up     = tile[(local_y + 1) * tile_width + local_x];
            float down   = tile[(local_y - 1) * tile_width + local_x];
            u_new[idx] = center + r * (left + right + up + down - 4.0f * center);
        }
    }
}

// External C wrapper
extern "C" void stencil_2d_optimized(
    float* u_new,
    const float* u_old,
    float r,
    int nx,
    int ny) {
    // Block and grid configuration
    dim3 block(32, 8);
    dim3 grid((ny + block.x - 1) / block.x,
              (nx + block.y - 1) / block.y);
    // Shared memory size: (blockDim.x+2) * (blockDim.y+2) * sizeof(float)
    size_t shared_bytes = (block.x + 2) * (block.y + 2) * sizeof(float);

    stencil_2d_kernel_optimized<<<grid, block, shared_bytes>>>(u_new, u_old, r, nx, ny);
    cudaDeviceSynchronize();
}