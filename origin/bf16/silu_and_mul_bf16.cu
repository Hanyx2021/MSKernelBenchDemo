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
#include <float.h>

__global__ void silu_and_mul_bf16_kernel(
    const __nv_bfloat16* in, __nv_bfloat16* out,
    int32_t B, int32_t D)
{
  for (int32_t b = blockIdx.x; b < B; b += gridDim.x) {
    const __nv_bfloat16* row = in + b * 2 * D;
    __nv_bfloat16* o = out + b * D;
    const __nv_bfloat16* x_ptr = row;
    const __nv_bfloat16* g_ptr = row + D;

    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      float xv = __bfloat162float(x_ptr[d]);
      float gv = __bfloat162float(g_ptr[d]);
      
      float silu_xv = xv / (1.0f + expf(-xv));
      
      o[d] = __float2bfloat16(silu_xv * gv);
    }
  }
}

extern "C" void silu_and_mul_bf16(
    const __nv_bfloat16* in, __nv_bfloat16* out,
    int32_t B, int32_t D) {
    dim3 block(128, 1, 1);
    dim3 grid(min(256, B), 1, 1);
    
    silu_and_mul_bf16_kernel<<<grid, block>>>(in, out, B, D);
    cudaDeviceSynchronize();
}