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

__global__ void merge_sort_bf16_kernel(__nv_bfloat16* data, __nv_bfloat16* temp, int N, int blockSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * 2 * blockSize;
    
    if (start >= N) return;
    
    int mid = min(start + blockSize, N);
    int end = min(start + 2 * blockSize, N);
    
    if (mid >= end) return;
    
    int i = start, j = mid, k = start;
    
    while (i < mid && j < end) {
        if (data[i] <= data[j]) {
            temp[k++] = data[i++];
        } else {
            temp[k++] = data[j++];
        }
    }
    while (i < mid) temp[k++] = data[i++];
    while (j < end) temp[k++] = data[j++];
}

extern "C" void sorting_bf16(__nv_bfloat16* data, int N) {
    __nv_bfloat16* d_temp;
    cudaMalloc(&d_temp, N * sizeof(__nv_bfloat16));
    cudaMemcpy(d_temp, data, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    
    for (int blockSize = 1; blockSize < N; blockSize *= 2) {
        int threadsNeeded = (N + 2 * blockSize - 1) / (2 * blockSize);
        int threadsPerBlock = 256;
        int blocks = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
        
        merge_sort_bf16_kernel<<<blocks, threadsPerBlock>>>(d_temp, data, N, blockSize);
        cudaDeviceSynchronize();

        merge_sort_bf16_kernel<<<blocks, threadsPerBlock>>>(data, d_temp, N, blockSize);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(data, d_temp, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    cudaFree(d_temp);
}