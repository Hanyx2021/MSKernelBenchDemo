#include <algorithm>
#include <climits>
#include <cmath>
#include <cmath>      // for INFINITY
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <optional>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <vector>


// Simple function to check if two bf16 values are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====
// Structure for value-index pair
struct ValueIndexPairBf16Optimized {
    __nv_bfloat16 value;
    int index;
};

// Convert bf16 to float for comparisons
__device__ inline float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Comparison: return true if a > b, or a == b and a.index < b.index
__device__ inline bool pair_greater(const ValueIndexPairBf16Optimized &a,
                                    const ValueIndexPairBf16Optimized &b) {
    float fa = bf16_to_float(a.value);
    float fb = bf16_to_float(b.value);
    if (fa > fb) return true;
    if (fa < fb) return false;
    // tie-break by smaller index
    return a.index < b.index;
}

// Phase 1: Hierarchical lock-free block-local top-k via warp-level reduction
__global__ void blockLocalTopK_kernel_optimized(
    const __nv_bfloat16* input,
    ValueIndexPairBf16Optimized* block_topk,
    int N, int k) {
    const int localCount = 4;  // each thread loads 4
    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;
    const int warpSize = 32;
    const int warpId = tid / warpSize;
    const int laneId = tid % warpSize;
    const int W = blockDimX / warpSize;  // warps per block
    int blockStart = blockIdx.x * blockDimX * localCount;

    // Shared buffers
    __shared__ ValueIndexPairBf16Optimized warpSubBuffer[1024];
    __shared__ ValueIndexPairBf16Optimized warpTopKBuffer[1024];

    // Each thread loads localCount elements
    ValueIndexPairBf16Optimized localArr[localCount];
    #pragma unroll
    for (int i = 0; i < localCount; ++i) {
        int idx = blockStart + tid + i * blockDimX;
        if (idx < N) {
            localArr[i].value = input[idx];
            localArr[i].index = idx;
        } else {
            localArr[i].value = __float2bfloat16(-INFINITY);
            localArr[i].index = INT_MAX;
        }
    }
    // sort localArr descending (small fixed size)
    #pragma unroll
    for (int i = 1; i < localCount; ++i) {
        auto key = localArr[i];
        int j = i - 1;
        while (j >= 0 && pair_greater(key, localArr[j])) {
            localArr[j + 1] = localArr[j]; j--;
        }
        localArr[j + 1] = key;
    }
    // write localArr into shared warpSubBuffer
    int warpBase = warpId * warpSize * localCount;
    #pragma unroll
    for (int i = 0; i < localCount; ++i) {
        warpSubBuffer[warpBase + laneId * localCount + i] = localArr[i];
    }
    __syncthreads();

    // Each warp leader builds warp-local top-k by scanning full segment
    int warpSegSize = warpSize * localCount;
    if (laneId == 0) {
        int topCount = k;
        ValueIndexPairBf16Optimized tmp[128];  // k <= 128
        int curCount = 0;
        for (int i = 0; i < warpSegSize; ++i) {
            auto v = warpSubBuffer[warpBase + i];
            if (curCount < topCount) {
                tmp[curCount++] = v;
                if (curCount == topCount) {
                    // initial insertion sort
                    for (int p = 1; p < topCount; ++p) {
                        auto key = tmp[p];
                        int q = p - 1;
                        while (q >= 0 && pair_greater(key, tmp[q])) {
                            tmp[q+1] = tmp[q]; q--;
                        }
                        tmp[q+1] = key;
                    }
                }
            } else if (pair_greater(v, tmp[topCount - 1])) {
                // replace smallest and re-insert
                tmp[topCount - 1] = v;
                int q = topCount - 2;
                while (q >= 0 && pair_greater(tmp[q+1], tmp[q])) {
                    auto sw = tmp[q]; tmp[q] = tmp[q+1]; tmp[q+1] = sw;
                    q--;
                }
            }
        }
        // write out this warp's top-k into warpTopKBuffer
        int warpKbase = warpId * k;
        for (int i = 0; i < topCount; ++i) {
            warpTopKBuffer[warpKbase + i] = tmp[i];
        }
    }
    __syncthreads();

    // Block-level merge: gather all W*k candidates
    __shared__ ValueIndexPairBf16Optimized mergeBuf[1024];
    int idxBuf = tid;
    int totalWarpCand = W * k;
    while (idxBuf < totalWarpCand) {
        mergeBuf[idxBuf] = warpTopKBuffer[idxBuf];
        idxBuf += blockDimX;
    }
    __syncthreads();

    // Simple single-threaded selection: pick top-k out of total candidates
    if (tid == 0) {
        for (int i = 0; i < k; ++i) {
            int best = i;
            for (int j = i + 1; j < totalWarpCand; ++j) {
                if (pair_greater(mergeBuf[j], mergeBuf[best])) {
                    best = j;
                }
            }
            auto tmp = mergeBuf[i];
            mergeBuf[i] = mergeBuf[best];
            mergeBuf[best] = tmp;
        }
    }
    __syncthreads();

    // write out the top-k into block_topk
    int outBase = blockIdx.x * k;
    int lane = tid % warpSize;
    for (int j = lane; j < k; j += warpSize) {
        block_topk[outBase + j] = mergeBuf[j];
    }
}

// Phase 2: Parallel Multi-Threaded Global Merge via Tree-Structured k-Way Reduction
__global__ void top_k_bf16_optimized(
    const ValueIndexPairBf16Optimized* block_topk,
    __nv_bfloat16* top_k_values,
    int* top_k_indices,
    int B, int k) {
    int lane = threadIdx.x;
    if (lane >= k) return;
    extern __shared__ ValueIndexPairBf16Optimized sh_mem[];
    // sh_mem[0..k-1] for S, sh_mem[k..2k-1] for buffer/T
    ValueIndexPairBf16Optimized* S   = sh_mem;
    ValueIndexPairBf16Optimized* buf = sh_mem + k;

    // Phase 0: load first block's k results into S
    S[lane] = block_topk[lane];
    __syncthreads();

    // Merge each subsequent block's top-k
    for (int i = 1; i < B; ++i) {
        // load block i's top-k list into buf
        buf[lane] = block_topk[i * k + lane];
        __syncthreads();
        
        // parallel merge of S and buf into buf
        // Each lane handles one element from S and one from buf
        ValueIndexPairBf16Optimized a = S[lane];
        // binary search in buf for number of elements > a
        int low = 0, high = k;
        while (low < high) {
            int mid = (low + high) >> 1;
            if (pair_greater(buf[mid], a)) low = mid + 1;
            else high = mid;
        }
        int posA = lane + low;

        ValueIndexPairBf16Optimized b = buf[lane];
        // binary search in S for number of elements > b
        low = 0; high = k;
        while (low < high) {
            int mid = (low + high) >> 1;
            if (pair_greater(S[mid], b)) low = mid + 1;
            else high = mid;
        }
        int posB = lane + low;

        // write merged results to buf as temporary
        if (posA < k) buf[posA] = a;
        if (posB < k) buf[posB] = b;
        __syncthreads();

        // copy buf back to S for next iteration
        S[lane] = buf[lane];
        __syncthreads();
    }

    // write final top-k results
    top_k_values[lane]  = S[lane].value;
    top_k_indices[lane] = S[lane].index;
}

extern "C" void top_k_bf16_optimized(
    __nv_bfloat16* top_k_values,
    int* top_k_indices,
    const __nv_bfloat16* input,
    const int N,
    const int k) {
    const int blockDimX = 256;
    const int localCount = 4;
    int chunkSize = blockDimX * localCount;
    int B = (N + chunkSize - 1) / chunkSize;

    // Allocate block-level top-k buffer
    ValueIndexPairBf16Optimized* d_block_topk;
    cudaMalloc(&d_block_topk, B * k * sizeof(ValueIndexPairBf16Optimized));

    // Launch block-local top-k kernel
    dim3 block(blockDimX);
    dim3 grid(B);
    blockLocalTopK_kernel_optimized<<<grid, block>>>(input, d_block_topk, N, k);
    cudaDeviceSynchronize();

    // Phase 2: parallel global merge
    int threads = 1;
    while (threads < k) threads <<= 1;
    if (threads > 1024) threads = 1024;
    size_t shared_mem = 2 * k * sizeof(ValueIndexPairBf16Optimized);
    top_k_bf16_optimized<<<1, threads, shared_mem>>>(
        d_block_topk, top_k_values, top_k_indices, B, k);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_block_topk);
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
    printf("Test case size: N: %d, K: %d. Complexity: %ld\n", test_case.N, test_case.K, (long) test_case.N * test_case.N);
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