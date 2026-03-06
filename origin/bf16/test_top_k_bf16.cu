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
#include <tuple>
#include <float.h>

// Simple function to check if two bf16 values are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====

struct ValueIndexPairBf16Optimized {
    __nv_bfloat16 value;
    int index;
};

__global__ void init_value_index_pairs_bf16_kernel_optimized(ValueIndexPairBf16Optimized* data, const __nv_bfloat16* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx].value = input[idx];
        data[idx].index = idx;
    }
}

__global__ void extract_top_k_bf16_kernel_optimized(const ValueIndexPairBf16Optimized* sorted_data, __nv_bfloat16* top_k_values, int* top_k_indices, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        top_k_values[idx] = sorted_data[idx].value;
        top_k_indices[idx] = sorted_data[idx].index;
    }
}

__global__ void merge_sort_kernel_with_indices_bf16_optimized(ValueIndexPairBf16Optimized* data, ValueIndexPairBf16Optimized* temp, int N, int blockSize) {
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

extern "C" void top_k_bf16_optimized(
    __nv_bfloat16* top_k_values,
    int* top_k_indices,
    const __nv_bfloat16* input,
    const int N,
    const int k)
{
    ValueIndexPairBf16Optimized* d_data;
    ValueIndexPairBf16Optimized* d_temp;
    cudaMalloc(&d_data, N * sizeof(ValueIndexPairBf16Optimized));
    cudaMalloc(&d_temp, N * sizeof(ValueIndexPairBf16Optimized));
    
    dim3 blockSize(256);
    dim3 gridSize((N + 255) / 256);
    
    init_value_index_pairs_bf16_kernel_optimized<<<gridSize, blockSize>>>(d_data, input, N);
    cudaDeviceSynchronize();
    
    for (int mergeBlockSize = 1; mergeBlockSize < N; mergeBlockSize *= 2) {
        int threadsNeeded = (N + 2 * mergeBlockSize - 1) / (2 * mergeBlockSize);
        int blocks = (threadsNeeded + 255) / 256;
        
        merge_sort_kernel_with_indices_bf16_optimized<<<blocks, blockSize>>>(d_data, d_temp, N, mergeBlockSize);
        cudaDeviceSynchronize();

        merge_sort_kernel_with_indices_bf16_optimized<<<blocks, blockSize>>>(d_temp, d_data, N, mergeBlockSize);
        cudaDeviceSynchronize();
    }
    
    dim3 extractGrid((k + 255) / 256);
    extract_top_k_bf16_kernel_optimized<<<extractGrid, blockSize>>>(d_data, top_k_values, top_k_indices, k);
    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    cudaFree(d_temp);
}

// ==== OPTIMIZED KERNEL END ====

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

extern "C" void top_k_bf16_origin(
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

// Test case input data structure
typedef struct {
    int N;
    int K;
    __nv_bfloat16 *input;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16};
    std::vector<int> K_list = {32, 64, 128};

    for (int i = 0; i < N_list.size(); i++) {
        for(int j = 0 ; j < K_list.size(); j++)
        {
            TestCase test_case;
            test_case.N = N_list[i];
            test_case.K = K_list[j];
            
            // Use fixed seed for reproducibility
            std::random_device rd;
            std::mt19937 rng(rd());  // Random seed for testing
            std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
            
            int input_item = test_case.N;
            test_case.input = new __nv_bfloat16[input_item];
            
            for (int ii = 0; ii < input_item; ii++) {
                test_case.input[ii] = __float2bfloat16(input_dist(rng));
            }
            test_case_list.push_back(test_case);
        }
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: N: %d, K: %d. Complexity: %ld\n", test_case.N, test_case.K, (long)(test_case.N * log2((double)test_case.N)));
}

// Function to warm up GPU and stabilize frequency
void stabilize_gpu() {
    // Create a dummy kernel to warm up GPU
    float *d_temp;
    cudaMalloc(&d_temp, sizeof(float));
    cudaFree(d_temp);
    
    // Small delay to let GPU stabilize
    for (volatile int i = 0; i < 10000; i++); // Busy wait
}

// Function to measure kernel performance with multiple iterations
template<typename KernelFunc>
float measure_kernel_performance(KernelFunc kernel, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        kernel();
    }
    
    // Measure multiple iterations
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return total_time ;  // Average time per iteration
}

// Function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    /* ================  Prepare data  ================ */
    std::vector<TestCase> test_case_list;
    load_test_case(test_case_list);

    for (const auto& test_case : test_case_list) {
        // Calculate sizes
        const int input_item = test_case.N;
        const int select_item = test_case.K;
        size_t input_size = input_item * sizeof(__nv_bfloat16);
        size_t output_size = select_item * sizeof(__nv_bfloat16);
        size_t index_size = select_item * sizeof(int);

        // Host memory inputs
        __nv_bfloat16* h_input = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(output_size);
        __nv_bfloat16* h_output_optimized = (__nv_bfloat16*)malloc(output_size);
        int* h_output_index = (int*)malloc(index_size);
        int* h_output_index_optimized = (int*)malloc(index_size);

        if (!h_input || !h_output || !h_output_optimized || !h_output_index  || !h_output_index_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_input, test_case.input, input_size);

        // GPU memory allocation
        __nv_bfloat16 *d_input, *d_output, *d_output_optimized;
        int *d_output_index, *d_output_index_optimized;

        checkCudaError(cudaMalloc((void**)&d_input, input_size), "Allocating d_input");
        checkCudaError(cudaMalloc((void**)&d_output, output_size), "Allocating d_output");
        checkCudaError(cudaMalloc((void**)&d_output_optimized, output_size), "Allocating d_output_optimized");
        checkCudaError(cudaMalloc((void**)&d_output_index, index_size), "Allocating d_output_index");
        checkCudaError(cudaMalloc((void**)&d_output_index_optimized, index_size), "Allocating d_output_index_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "Copying h_input to d_input");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            top_k_bf16_origin(d_output, d_output_index, d_input, test_case.N, test_case.K);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            top_k_bf16_optimized(d_output_optimized, d_output_index_optimized, d_input, test_case.N, test_case.K);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
        checkCudaError(cudaMemcpy(h_output_optimized, d_output_optimized, output_size, cudaMemcpyDeviceToHost), "Copying d_output_optimized to h_output_optimized");
        checkCudaError(cudaMemcpy(h_output_index, d_output_index, index_size, cudaMemcpyDeviceToHost), "Copying d_output_index to h_output_index");
        checkCudaError(cudaMemcpy(h_output_index_optimized, d_output_index_optimized, index_size, cudaMemcpyDeviceToHost), "Copying d_output_index_optimized to h_output_index_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < select_item; i++) {
            if (!bfloat16_equals(h_output[i], h_output_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_output[i]), __bfloat162float(h_output_optimized[i]));
                return 1;
            }
        }

        for (int i = 0; i < select_item; i++) {
            if (h_output_index[i] != h_output_index_optimized[i]) {
                printf("Output index mismatch at index %d: original %d, optimized %d\n", i, h_output_index[i], h_output_index_optimized[i]);
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_output);
        cudaFree(d_output_optimized);
        cudaFree(d_output_index);
        cudaFree(d_output_index_optimized);

        free(h_output);
        free(h_output_optimized);
        free(h_output_index);
        free(h_output_index_optimized);
        delete [] test_case.input;
    }

    return 0;
}