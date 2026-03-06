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
#include <cub/cub.cuh>

struct ValueIndexPair {
    float value;
    int index;
};

__global__ void init_value_index_pairs_kernel(ValueIndexPair* data, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx].value = input[idx];
        data[idx].index = idx;
    }
}

__global__ void extract_top_k_kernel(const ValueIndexPair* sorted_data, float* top_k_values, int* top_k_indices, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        top_k_values[idx] = sorted_data[idx].value;
        top_k_indices[idx] = sorted_data[idx].index;
    }
}

__global__ void merge_sort_kernel_with_indices(ValueIndexPair* data, ValueIndexPair* temp, int N, int blockSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * 2 * blockSize;
    
    if (start >= N) return;
    
    int mid = min(start + blockSize, N);
    int end = min(start + 2 * blockSize, N);
    
    if (mid >= end) return;
    
    int i = start, j = mid, k_idx = start;
    
    while (i < mid && j < end) {
        if (data[i].value > data[j].value) {
            temp[k_idx++] = data[i++];
        } 
        else if (data[i].value < data[j].value) {
            temp[k_idx++] = data[j++];
        }
        else {
            if (data[i].index < data[j].index) {
                temp[k_idx++] = data[i++];
            } else {
                temp[k_idx++] = data[j++];
            }
        }
    }
    while (i < mid) temp[k_idx++] = data[i++];
    while (j < end) temp[k_idx++] = data[j++];
}

extern "C" void top_k(
    float* top_k_values,
    int* top_k_indices,
    const float* input,
    const int N,
    const int k)
{
    ValueIndexPair* d_data;
    ValueIndexPair* d_temp;
    cudaMalloc(&d_data, N * sizeof(ValueIndexPair));
    cudaMalloc(&d_temp, N * sizeof(ValueIndexPair));
    
    dim3 blockSize(256);
    dim3 gridSize((N + 255) / 256);
    
    init_value_index_pairs_kernel<<<gridSize, blockSize>>>(d_data, input, N);
    cudaDeviceSynchronize();
    
    for (int mergeBlockSize = 1; mergeBlockSize < N; mergeBlockSize *= 2) {
        int threadsNeeded = (N + 2 * mergeBlockSize - 1) / (2 * mergeBlockSize);
        int blocks = (threadsNeeded + 255) / 256;
        
        merge_sort_kernel_with_indices<<<blocks, blockSize>>>(d_data, d_temp, N, mergeBlockSize);
        cudaDeviceSynchronize();

        merge_sort_kernel_with_indices<<<blocks, blockSize>>>(d_temp, d_data, N, mergeBlockSize);
        cudaDeviceSynchronize();
    }
    
    dim3 extractGrid((k + 255) / 256);
    extract_top_k_kernel<<<extractGrid, blockSize>>>(d_data, top_k_values, top_k_indices, k);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    cudaFree(d_temp);
}