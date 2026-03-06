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


__global__ void row_tanh_kernel(
    float* out,
    const float* input,
    const int N) {
  for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
      const float x = input[idx];
      out[idx] = tanhf(x);
  }
}

extern "C" void row_tanh(
    float* out,
    const float* input,
    const int N) {
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    row_tanh_kernel<<<grid, block>>>(out, input, N);
    cudaDeviceSynchronize();
}