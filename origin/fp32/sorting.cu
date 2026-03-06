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

__global__ void merge_sort_kernel(float* data, float* temp, int N, int blockSize) {
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

extern "C" void sorting(float* data, int N) {
    float* d_temp;
    cudaMalloc(&d_temp, N * sizeof(float));
    cudaMemcpy(d_temp, data, N * sizeof(float), cudaMemcpyDeviceToDevice);
    
    for (int blockSize = 1; blockSize < N; blockSize *= 2) {
        int threadsNeeded = (N + 2 * blockSize - 1) / (2 * blockSize);
        int threadsPerBlock = 256;
        int blocks = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
        
        merge_sort_kernel<<<blocks, threadsPerBlock>>>(d_temp, data, N, blockSize);
        cudaDeviceSynchronize();

        merge_sort_kernel<<<blocks, threadsPerBlock>>>(data, d_temp, N, blockSize);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(data, d_temp, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_temp);
}