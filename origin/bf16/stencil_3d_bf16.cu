#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

__global__ void stencil_3d_bf16_kernel(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny, int nz) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= 1 && i < nx-1 && 
        j >= 1 && j < ny-1 && 
        k >= 1 && k < nz-1) {
        
        int idx = i * ny * nz + j * nz + k;
        
        float center = __bfloat162float(u_old[idx]);
        
        float left   = __bfloat162float(u_old[idx - ny*nz]);
        float right  = __bfloat162float(u_old[idx + ny*nz]);

        float up     = __bfloat162float(u_old[idx + nz]);
        float down   = __bfloat162float(u_old[idx - nz]);
        
        float front  = __bfloat162float(u_old[idx + 1]);
        float back   = __bfloat162float(u_old[idx - 1]);

        u_new[idx] = __float2bfloat16(center + r * (left + right + up + down + front + back - 6.0f * center));
    }
}

__global__ void copy_boundary_3d_bf16_kernel(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
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

extern "C" void stencil_3d_bf16(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny, int nz) {
    
    dim3 block(4, 4, 4);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );
    
    stencil_3d_bf16_kernel<<<grid, block>>>(u_new, u_old, r, nx, ny, nz);

    copy_boundary_3d_bf16_kernel<<<grid, block>>>(u_new, u_old, nx, ny, nz);
    
    cudaDeviceSynchronize();
}