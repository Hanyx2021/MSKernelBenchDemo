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
#include <cub/cub.cuh>

struct ValueIndexPairBf16 {
    __nv_bfloat16 value;
    int index;
};

__global__ void init_value_index_pairs_bf16_kernel(ValueIndexPairBf16* data, const __nv_bfloat16* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx].value = input[idx];
        data[idx].index = idx;
    }
}

__global__ void extract_top_k_bf16_kernel(const ValueIndexPairBf16* sorted_data, __nv_bfloat16* top_k_values, int* top_k_indices, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        top_k_values[idx] = sorted_data[idx].value;
        top_k_indices[idx] = sorted_data[idx].index;
    }
}

__global__ void merge_sort_kernel_with_indices_bf16(ValueIndexPairBf16* data, ValueIndexPairBf16* temp, int N, int blockSize) {
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

extern "C" void top_k_bf16(
    __nv_bfloat16* top_k_values,
    int* top_k_indices,
    const __nv_bfloat16* input,
    const int N,
    const int k)
{
    ValueIndexPairBf16* d_data;
    ValueIndexPairBf16* d_temp;
    cudaMalloc(&d_data, N * sizeof(ValueIndexPairBf16));
    cudaMalloc(&d_temp, N * sizeof(ValueIndexPairBf16));
    
    dim3 blockSize(256);
    dim3 gridSize((N + 255) / 256);
    
    init_value_index_pairs_bf16_kernel<<<gridSize, blockSize>>>(d_data, input, N);
    cudaDeviceSynchronize();
    
    for (int mergeBlockSize = 1; mergeBlockSize < N; mergeBlockSize *= 2) {
        int threadsNeeded = (N + 2 * mergeBlockSize - 1) / (2 * mergeBlockSize);
        int blocks = (threadsNeeded + 255) / 256;
        
        merge_sort_kernel_with_indices_bf16<<<blocks, blockSize>>>(d_data, d_temp, N, mergeBlockSize);
        cudaDeviceSynchronize();

        merge_sort_kernel_with_indices_bf16<<<blocks, blockSize>>>(d_temp, d_data, N, mergeBlockSize);
        cudaDeviceSynchronize();
    }
    
    dim3 extractGrid((k + 255) / 256);
    extract_top_k_bf16_kernel<<<extractGrid, blockSize>>>(d_data, top_k_values, top_k_indices, k);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    cudaFree(d_temp);
}