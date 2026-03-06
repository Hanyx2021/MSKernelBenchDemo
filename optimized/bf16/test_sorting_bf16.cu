#include <algorithm>
#include <cmath>
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
// Small chunk size for initial bitonic sort
#define CHUNK 1024

// Device parameters for dynamic partitioning
#define SM_COUNT 128       // Number of SMs on target device
#define BSM 8              // Desired blocks per SM
#define TARGET_BLOCKS (SM_COUNT * BSM)

// Merge kernel thread configuration
#define MERGE_THREADS_OPT 256
#define WARP_SIZE_OPT 32

// Utility for integer division with rounding up
inline int iDivUp(int a, int b) {
    return (a + b - 1) / b;
}

// In-block bitonic sort for small chunks using shared memory (one thread per element)
__global__ void bitonic_sort_bf16_kernel_optimized(__nv_bfloat16* data, int N) {
    extern __shared__ float sdata[];      // CHUNK floats per block
    int tid = threadIdx.x;
    int base = blockIdx.x * CHUNK;
    int idx = base + tid;

    // load one element per thread
    float v = (idx < N)
              ? __bfloat162float(data[idx])
              : FLT_MAX;
    sdata[tid] = v;
    __syncthreads();

    // classic bitonic network over CHUNK elements
    for (int k = 2; k <= CHUNK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid && ixj < CHUNK) {
                bool up = ((tid & k) == 0);
                float a = sdata[tid];
                float b = sdata[ixj];
                // when up==true we want a<=b, when up==false we want a>=b
                if ((a > b) == up) {
                    sdata[tid]  = b;
                    sdata[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // write back sorted data to global memory
    if (idx < N) {
        data[idx] = __float2bfloat16(sdata[tid]);
    }
}

// Partitioned merge kernel with dynamic partitions per pass
__global__ void merge_partitioned_bf16_kernel_optimized(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int N,
    int blockSize,
    int partitions)
{
    int global_block = blockIdx.x;
    int part_idx = global_block % partitions;
    int run_idx = global_block / partitions;

    int start = run_idx * 2 * blockSize;
    if (start >= N) return;
    int mid = start + blockSize;
    mid = (mid < N) ? mid : N;
    int end = start + 2 * blockSize;
    end = (end < N) ? end : N;

    int A_size = mid - start;
    int B_size = end - mid;
    int total = A_size + B_size;

    // Partition length
    int L = (total + partitions - 1) / partitions;
    int local_start = part_idx * L;
    if (local_start >= total) return;
    int local_L = (local_start + L < total) ? L : (total - local_start);

    int tid = threadIdx.x;
    if (tid >= local_L) return;

    int k = local_start + tid;
    // Merge path binary search to find split
    int low = k - B_size;
    if (low < 0) low = 0;
    int high = (k < A_size) ? k : A_size;
    while (low < high) {
        int midp = (low + high) >> 1;
        float aval = __bfloat162float(input[start + midp]);
        int bidx = k - midp - 1;
        float bval = FLT_MAX;
        if (bidx >= 0 && bidx < B_size) {
            bval = __bfloat162float(input[mid + bidx]);
        }
        if (aval <= bval) {
            low = midp + 1;
        } else {
            high = midp;
        }
    }
    int a_j = low;
    int b_j = k - a_j;

    float valA = FLT_MAX;
    if (a_j < A_size) {
        valA = __bfloat162float(input[start + a_j]);
    }
    float valB = FLT_MAX;
    if (b_j < B_size) {
        valB = __bfloat162float(input[mid + b_j]);
    }
    float v = (valA <= valB) ? valA : valB;

    output[start + k] = __float2bfloat16(v);
}

extern "C" void sorting_bf16_optimized(__nv_bfloat16* data, int N) {
    // Allocate temporary buffer
    __nv_bfloat16* d_temp;
    cudaMalloc(&d_temp, N * sizeof(__nv_bfloat16));
    cudaMemcpy(d_temp, data, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

    // 1) In-block bitonic sort on chunks
    int blocksChunk = iDivUp(N, CHUNK);
    bitonic_sort_bf16_kernel_optimized<<<blocksChunk, CHUNK, CHUNK * sizeof(float)>>>(data, N);

    // Copy sorted chunks to temp buffer
    cudaMemcpy(d_temp, data, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

    // 2) Global merge passes with dynamic partitions (ping-pong)
    const __nv_bfloat16* in = d_temp;
    __nv_bfloat16* out = data;
    for (int blockSize = CHUNK; blockSize < N; blockSize <<= 1) {
        int runPairs = iDivUp(N, 2 * blockSize);
        int partitions = iDivUp(TARGET_BLOCKS, runPairs);
        if (partitions < 1) partitions = 1;
        int maxAllowed = (2 * blockSize + WARP_SIZE_OPT - 1) / WARP_SIZE_OPT;
        if (maxAllowed < 1) maxAllowed = 1;
        if (partitions > maxAllowed) partitions = maxAllowed;

        int grid = runPairs * partitions;
        merge_partitioned_bf16_kernel_optimized<<<grid, MERGE_THREADS_OPT>>>(in, out, N, blockSize, partitions);

        // swap buffers
        const __nv_bfloat16* tmp = in;
        in = out;
        out = const_cast<__nv_bfloat16*>(tmp);
    }

    // If final result is not in "data", copy it back
    if (in != data) {
        cudaMemcpy(data, in, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    }
    cudaFree(d_temp);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

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

extern "C" void sorting_bf16_origin(__nv_bfloat16* data, int N) {
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

// Test case input data structure
typedef struct {
    int N;
    __nv_bfloat16 *input;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16};

    for (int i = 0; i < N_list.size(); i++) {
        TestCase test_case;
        test_case.N = N_list[i];
        
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

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: N: %d. Complexity: %ld\n", test_case.N, (long)test_case.N * test_case.N);
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
        size_t data_size = input_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(data_size);
        __nv_bfloat16* h_output_optimized = (__nv_bfloat16*)malloc(data_size);

        if (!h_output || !h_output_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_output, test_case.input, data_size);
        memcpy(h_output_optimized, test_case.input, data_size);

        // GPU memory allocation
        __nv_bfloat16 *d_output, *d_output_optimized;

        checkCudaError(cudaMalloc((void**)&d_output, data_size), "Allocating d_output");
        checkCudaError(cudaMalloc((void**)&d_output_optimized, data_size), "Allocating d_output_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_output, h_output, data_size, cudaMemcpyHostToDevice), "Copying h_output to d_output");
        checkCudaError(cudaMemcpy(d_output_optimized, h_output_optimized, data_size, cudaMemcpyHostToDevice), "Copying h_output_optimized to d_output_optimized");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            sorting_bf16_origin(d_output, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            sorting_bf16_optimized(d_output_optimized, test_case.N);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
        checkCudaError(cudaMemcpy(h_output_optimized, d_output_optimized, data_size, cudaMemcpyDeviceToHost), "Copying d_output_optimized to h_output_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < input_item; i++) {
            if (!bfloat16_equals(h_output[i], h_output_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_output[i]), __bfloat162float(h_output_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_output);
        cudaFree(d_output_optimized);

        free(h_output);
        free(h_output_optimized);
        delete [] test_case.input;
    }

    return 0;
}