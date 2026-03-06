#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <optional>
#include <algorithm>
#include <random>
#include <cmath>
#include <vector>
#include <functional>

__global__ void stencil_3d_kernel(
    float* u_new, 
    const float* u_old, 
    float r, int nx, int ny, int nz) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= 1 && i < nx-1 && 
        j >= 1 && j < ny-1 && 
        k >= 1 && k < nz-1) {
        
        int idx = i * ny * nz + j * nz + k;
        
        float center = u_old[idx];
        
        float left   = u_old[idx - ny*nz];
        float right  = u_old[idx + ny*nz];

        float up     = u_old[idx + nz];
        float down   = u_old[idx - nz];
        
        float front  = u_old[idx + 1];
        float back   = u_old[idx - 1];

        u_new[idx] = center + r * (
            left + right + up + down + front + back - 6.0f * center
        );
    }
}

__global__ void copy_boundary_3d_kernel(
    float* dst,
    const float* src,
    int nx, int ny, int nz) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz) {
        bool is_boundary = 
            (i == 0) || (i == nx - 1) ||
            (j == 0) || (j == ny - 1) ||
            (k == 0) || (k == nz - 1);
        
        if (is_boundary) {
            int idx = i * ny * nz + j * nz + k;
            dst[idx] = src[idx];
        }
    }
}

extern "C" void stencil_3d(
    float* u_new, 
    const float* u_old, 
    float r, int nx, int ny, int nz) {
    
    dim3 block(4, 4, 4);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );
    
    stencil_3d_kernel<<<grid, block>>>(u_new, u_old, r, nx, ny, nz);

    copy_boundary_3d_kernel<<<grid, block>>>(u_new, u_old, nx, ny, nz);
    
    cudaDeviceSynchronize();
}