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
#include <float.h>

__global__ void silu_and_mul_kernel(
    const float* in, float* out,
    int32_t B, int32_t D)
{
  for (int32_t b = blockIdx.x; b < B; b += gridDim.x) {
    const float* row = in + b * 2 * D;
    float* o = out + b * D;
    const float* x_ptr = row;
    const float* g_ptr = row + D;

    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      float xv = x_ptr[d];
      float gv = g_ptr[d];
      float silu_xv = xv / (1.0f + expf(-xv));
      o[d] = silu_xv * gv;
    }
  }
}

extern "C" void silu_and_mul(
    const float* in, float* out,
    int32_t B, int32_t D) {
    dim3 block(128, 1, 1);
    dim3 grid(min(256, B), 1, 1);
    
    silu_and_mul_kernel<<<grid, block>>>(in, out, B, D);
    cudaDeviceSynchronize();
}