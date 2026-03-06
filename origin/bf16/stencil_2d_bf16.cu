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

__global__ void stencil_2d_bf16_kernel(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1) {
        int idx = i * ny + j;

        float center = __bfloat162float(u_old[idx]);
        float left = __bfloat162float(u_old[idx - 1]);
        float right = __bfloat162float(u_old[idx + 1]);
        float up = __bfloat162float(u_old[idx + ny]);
        float down = __bfloat162float(u_old[idx - ny]);
        
        u_new[idx] = __float2bfloat16(center + r * (left + right + up + down - 4.0f * center));
    }
}

__global__ void copy_boundary_bf16_kernel(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            dst[i * ny + j] = src[i * ny + j];
        }
    }
}

extern "C" void stencil_2d_bf16(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny) {
    
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);
    
    stencil_2d_kernel<<<grid, block>>>(u_new, u_old, r, nx, ny);

    copy_boundary_kernel<<<grid, block>>>(u_new, u_old, nx, ny);

    cudaDeviceSynchronize();
}
