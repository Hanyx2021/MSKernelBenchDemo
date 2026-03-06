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

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void vector_add(const float* A, const float* B, float* C, int N) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    vector_add_kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();
}