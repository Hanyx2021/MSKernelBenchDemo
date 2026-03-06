#include <cuda.h>
#include <cuda_runtime.h>

// Optimized 3D stencil kernel with improved memory coalescing and read-only caching
__global__ void stencil_3d_kernel_optimized(
    float* __restrict__ u_new,
    const float* __restrict__ u_old,
    float r, int nx, int ny, int nz) {
    // Thread mapping: x -> k (fastest), y -> j, z -> i
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
        // Compute linear index
        int idx = i * (ny * nz) + j * nz + k;
        // Load through read-only cache
        float center = __ldg(&u_old[idx]);
        float left   = __ldg(&u_old[idx - (ny * nz)]);
        float right  = __ldg(&u_old[idx + (ny * nz)]);
        float up     = __ldg(&u_old[idx + nz]);
        float down   = __ldg(&u_old[idx - nz]);
        float front  = __ldg(&u_old[idx + 1]);
        float back   = __ldg(&u_old[idx - 1]);

        u_new[idx] = center + r * (left + right + up + down + front + back - 6.0f * center);
    }
}

// Optimized boundary copy kernel
__global__ void copy_boundary_3d_kernel_optimized(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int nx, int ny, int nz) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        bool is_boundary = (i == 0) || (i == nx - 1) || (j == 0) || (j == ny - 1) || (k == 0) || (k == nz - 1);
        if (is_boundary) {
            int idx = i * (ny * nz) + j * nz + k;
            dst[idx] = __ldg(&src[idx]);
        }
    }
}

// External C wrapper for optimized 3D stencil
extern "C" void stencil_3d_optimized(
    float* u_new,
    const float* u_old,
    float r, int nx, int ny, int nz) {
    // Define block and grid dimensions
    dim3 block(32, 4, 4);
    dim3 grid(
        (unsigned int)(nz + block.x - 1) / block.x,
        (unsigned int)(ny + block.y - 1) / block.y,
        (unsigned int)(nx + block.z - 1) / block.z
    );

    // Launch optimized stencil kernel
    stencil_3d_kernel_optimized<<<grid, block>>>(u_new, u_old, r, nx, ny, nz);
    // Launch optimized boundary copy kernel
    copy_boundary_3d_kernel_optimized<<<grid, block>>>(u_new, u_old, nx, ny, nz);

    // Synchronize
    cudaDeviceSynchronize();
}
