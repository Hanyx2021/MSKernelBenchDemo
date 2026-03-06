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
// Warp-level reduction utility
template <unsigned int W>
__inline__ __device__ float warp_reduce_sum_Optimized(float val) {
    #pragma unroll
    for (int offset = W >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Monte Carlo kernel: each block computes a partial sum and writes to block_sums
__global__ void monte_carlo_int_bf16_kernel_optimized(
    float* block_sums,
    const __nv_bfloat16* __restrict__ y_samples,
    int N) {
    extern __shared__ float warp_sums[]; // one float per warp

    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Double-buffered prefetch + unrolled loop
    float sum = 0.0f;
    int idx = tid_global;
    float reg0 = 0.0f;
    if (idx < N) {
        reg0 = __bfloat162float(__ldg(y_samples + idx));
        idx += stride;
    }
    #pragma unroll 8
    while (idx < N) {
        float reg1 = __bfloat162float(__ldg(y_samples + idx));
        sum += reg0;
        reg0 = reg1;
        idx += stride;
    }
    sum += reg0;

    // Warp-level reduction
    unsigned int lane = threadIdx.x & (warpSize - 1);
    unsigned int wid = threadIdx.x / warpSize;
    float wsum = warp_reduce_sum_Optimized<32>(sum);
    if (lane == 0) {
        warp_sums[wid] = wsum;
    }
    __syncthreads();

    // Block-level reduction by first warp
    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        int numWarps = blockDim.x / warpSize;
        for (int i = 0; i < numWarps; ++i) {
            block_sum += warp_sums[i];
        }
        block_sums[blockIdx.x] = block_sum;
    }
}

// Reduction & finalize kernel: reduce block_sums and write final bfloat16 result
__global__ void reduce_block_sums_kernel_optimized(
    const float* block_sums,
    __nv_bfloat16* result,
    float a,
    float b,
    int N,
    int numBlocks) {
    extern __shared__ float warp_sums[]; // one float per warp

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load and partial reduce
    float sum = 0.0f;
    for (int i = tid; i < numBlocks; i += stride) {
        sum += block_sums[i];
    }

    // Warp-level reduction
    unsigned int lane = tid & (warpSize - 1);
    unsigned int wid = tid / warpSize;
    float wsum = warp_reduce_sum_Optimized<32>(sum);
    if (lane == 0) {
        warp_sums[wid] = wsum;
    }
    __syncthreads();

    // Final reduction by first warp
    float total = 0.0f;
    if (wid == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        if (lane < numWarps) {
            total = warp_sums[lane];
        }
        total = warp_reduce_sum_Optimized<32>(total);
    }
    __syncthreads();

    // Thread 0 writes result
    if (tid == 0) {
        float integral = ((b - a) / static_cast<float>(N)) * total;
        *result = __float2bfloat16_rn(integral);
    }
}

extern "C" void monte_carlo_int_bf16_optimized(
    const __nv_bfloat16* y_samples,
    __nv_bfloat16* result,
    float a,
    float b,
    int N) {
    // Determine grid and block sizes
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    blocks = std::min(blocks, 1024);

    // Allocate per-block sums
    float* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, blocks * sizeof(float));

    // Launch kernel1: compute per-block sums
    size_t sharedMem1 = (threadsPerBlock / 32) * sizeof(float);
    monte_carlo_int_bf16_kernel_optimized<<<blocks, threadsPerBlock, sharedMem1>>>(
        d_block_sums, y_samples, N);

    // Launch kernel2: reduce block_sums and finalize
    const int threads2 = 256;
    size_t sharedMem2 = (threads2 / 32) * sizeof(float);
    reduce_block_sums_kernel_optimized<<<1, threads2, sharedMem2>>>(
        d_block_sums, result, a, b, N, blocks);

    // Synchronize and cleanup
    cudaDeviceSynchronize();
    cudaFree(d_block_sums);
}
// ==== OPTIMIZED KERNEL END ====

__global__ void monte_carlo_int_bf16_kernel(
    float* integral_sum,
    const __nv_bfloat16* y_samples,
    float a,
    float b,
    int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    
    for (int i = tid; i < N; i += stride) {
        float y_val = __bfloat162float(y_samples[i]);
        local_sum += y_val;
    }
    
    if (local_sum != 0.0f) {
        atomicAdd(integral_sum, local_sum);
    }
}

__global__ void finalize_integral_kernel(
    float* integral_sum,
    __nv_bfloat16* result,
    float a,
    float b,
    int N) {

    float avg = *integral_sum / static_cast<float>(N);
    float integral_value = (b - a) * avg;

    *result = __float2bfloat16_rn(integral_value);
}

extern "C" void monte_carlo_int_bf16_origin(
    const __nv_bfloat16* y_samples,
    __nv_bfloat16* result,
    float a,
    float b,
    int N) {

    float* d_integral_sum;
    cudaMalloc(&d_integral_sum, sizeof(float));
    cudaMemset(d_integral_sum, 0, sizeof(float));

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;

    monte_carlo_int_bf16_kernel<<<blocks, threadsPerBlock>>>(
        d_integral_sum, y_samples, a, b, N);

    finalize_integral_kernel<<<1, 1>>>(
        d_integral_sum, result, a, b, N);

    cudaDeviceSynchronize();
    cudaFree(d_integral_sum);
}

// Test case input data structure
typedef struct {
    int N;
    float a;
    float b;
    __nv_bfloat16 *y_samples;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};

    for (int i = 0; i < N_list.size(); i++) {
        TestCase test_case;
        test_case.N = N_list[i];
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-10.0f, 10.0f);

        std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
        
        float rand1 = dist(rng);
        float rand2 = dist(rng);
        
        test_case.a = std::min(rand1, rand2);
        test_case.b = std::max(rand1, rand2);
        
        int item_count = test_case.N;
        test_case.y_samples = new __nv_bfloat16[item_count];
        
        // Generate random probabilities and normalize them
        for (int ii = 0; ii < item_count; ii++) {
            test_case.y_samples[ii] = __float2bfloat16(input_dist(rng));
        }
        
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: N: %d. Complexity: %d\n", test_case.N, test_case.N);
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
        const int item_count = test_case.N;
        size_t data_size = item_count * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_y_samples = (__nv_bfloat16*)malloc(data_size);
        __nv_bfloat16 h_result_origin, h_result_optimized;

        if (!h_y_samples) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_y_samples, test_case.y_samples, data_size);

        // GPU memory allocation
        __nv_bfloat16 *d_y_samples, *d_result_origin, *d_result_optimized;

        checkCudaError(cudaMalloc((void**)&d_y_samples, data_size), "Allocating d_y_samples");
        checkCudaError(cudaMalloc((void**)&d_result_origin, sizeof(__nv_bfloat16)), "Allocating d_result_origin");
        checkCudaError(cudaMalloc((void**)&d_result_optimized, sizeof(__nv_bfloat16)), "Allocating d_result_optimized");

        // Initialize loss values to zero
        __nv_bfloat16 zero = __float2bfloat16(0.0f);
        checkCudaError(cudaMemcpy(d_result_origin, &zero, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "Initializing d_result_origin");
        checkCudaError(cudaMemcpy(d_result_optimized, &zero, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "Initializing d_result_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_y_samples, h_y_samples, data_size, cudaMemcpyHostToDevice), "Copying h_y_samples to d_y_samples");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            monte_carlo_int_bf16_origin(d_y_samples, d_result_origin, test_case.a, test_case.b, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        auto optimized_kernel = [&]() {
            monte_carlo_int_bf16_optimized(d_y_samples, d_result_optimized, test_case.a, test_case.b, test_case.N);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(&h_result_origin, d_result_origin, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "Copying d_result_origin to host");
        checkCudaError(cudaMemcpy(&h_result_optimized, d_result_optimized, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "Copying d_result_optimized to host");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        if (!bfloat16_equals(h_result_origin, h_result_optimized, 1e-2f)) {
            printf("Output mismatch: original %.6f, optimized %.6f\n", __bfloat162float(h_result_origin), __bfloat162float(h_result_optimized));
            return 1;
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_y_samples);
        cudaFree(d_result_origin);
        cudaFree(d_result_optimized);

        free(h_y_samples);
        delete [] test_case.y_samples;
    }

    return 0;
}