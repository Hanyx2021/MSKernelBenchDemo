#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>
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


// Simple function to check if two floats are approximately equal
bool float_equals(float a, float b, float tolerance) {
    return fabs(a - b) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====
// Configuration for optimized kernels
struct MSE_loss_bf16_OptimizedConfig {
    static constexpr int threadsPerBlock = 256;
    static constexpr int maxBlocks = 1024;
};

// Stage 1: Per-block partial sum computation and reduction
__global__ void MSE_loss_bf16_stage1_kernel_optimized(
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    int N,
    float* partial_sums) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    unsigned int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    // Compute strided local sum
    for (int i = idx; i < N; i += stride) {
        float x = __bfloat162float(X[i]);
        float y = __bfloat162float(Y[i]);
        float d = x - y;
        local_sum += d * d;
    }
    // Store in shared memory
    sdata[tid] = local_sum;
    __syncthreads();
    // In-block reduction
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Stage 2: Reduce partial sums and perform final division
__global__ void MSE_loss_bf16_stage2_kernel_optimized(
    float* loss,
    const float* partial_sums,
    int numBlocks,
    int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    // Load partial sums
    if (tid < numBlocks) {
        sdata[tid] = partial_sums[tid];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    // Reduction
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    // Final division and write out
    if (tid == 0) {
        loss[0] = sdata[0] / static_cast<float>(N);
    }
}

// External C wrapper for optimized operator
extern "C" void MSE_loss_bf16_optimized(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {
    // Configure stage 1
    int threads1 = MSE_loss_bf16_OptimizedConfig::threadsPerBlock;
    int blocks = (N + threads1 - 1) / threads1;
    blocks = (blocks > MSE_loss_bf16_OptimizedConfig::maxBlocks)
             ? MSE_loss_bf16_OptimizedConfig::maxBlocks
             : blocks;
    // Allocate temporary partial sums
    float* partial_sums = nullptr;
    cudaMalloc(&partial_sums, blocks * sizeof(float));
    // Launch stage 1
    size_t sharedMem1 = threads1 * sizeof(float);
    MSE_loss_bf16_stage1_kernel_optimized<<<blocks, threads1, sharedMem1>>>(
        X, Y, N, partial_sums);
    // Configure stage 2
    int threads2 = 1;
    while (threads2 < blocks) threads2 <<= 1;
    if (threads2 > MSE_loss_bf16_OptimizedConfig::maxBlocks) {
        threads2 = MSE_loss_bf16_OptimizedConfig::maxBlocks;
    }
    size_t sharedMem2 = threads2 * sizeof(float);
    // Launch stage 2 (reduction + division)
    MSE_loss_bf16_stage2_kernel_optimized<<<1, threads2, sharedMem2>>>(
        loss, partial_sums, blocks, N);
    // Clean up
    cudaFree(partial_sums);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void MSE_loss_bf16_kernel(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    
    for (int i = tid; i < N; i += stride) {
        float x_val = __bfloat162float(X[i]);
        float y_val = __bfloat162float(Y[i]);
        float diff = x_val - y_val;
        local_sum += diff * diff;
    }
    
    if (local_sum != 0.0f) {
        atomicAdd(loss, local_sum);
    }
}

__global__ void divide_kernel_origin(float* value, float divisor) {
    *value = *value / divisor;
}

extern "C" void MSE_loss_bf16_origin(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {

    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;

    MSE_loss_bf16_kernel<<<blocks, threadsPerBlock>>>(loss, X, Y, N);

    divide_kernel_origin<<<1, 1>>>(loss, static_cast<float>(N));

    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int N;
    __nv_bfloat16 *X;
    __nv_bfloat16 *Y;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};

    for (int i = 0; i < N_list.size(); i++) {
        TestCase test_case;
        test_case.N = N_list[i];  // Larger size for better timing
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int item_count = test_case.N;
        test_case.X = new __nv_bfloat16[item_count];
        test_case.Y = new __nv_bfloat16[item_count];
        
        for (int ii = 0; ii < item_count; ii++) {
            test_case.X[ii] = __float2bfloat16(input_dist(rng));
        }
        for (int ii = 0; ii < item_count; ii++) {
            test_case.Y[ii] = __float2bfloat16(input_dist(rng));
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
        __nv_bfloat16* h_X = (__nv_bfloat16*)malloc(data_size);
        __nv_bfloat16* h_Y = (__nv_bfloat16*)malloc(data_size);
        float h_loss_original, h_loss_optimized;

        if (!h_X || !h_Y) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_X, test_case.X, data_size);
        memcpy(h_Y, test_case.Y, data_size);

        // GPU memory allocation
        __nv_bfloat16 *d_X, *d_Y;
        float *d_loss, *d_loss_optimized;

        checkCudaError(cudaMalloc((void**)&d_X, data_size), "Allocating d_X");
        checkCudaError(cudaMalloc((void**)&d_Y, data_size), "Allocating d_Y");
        checkCudaError(cudaMalloc((void**)&d_loss, sizeof(float)), "Allocating d_loss");
        checkCudaError(cudaMalloc((void**)&d_loss_optimized, sizeof(float)), "Allocating d_loss_optimized");

        // Initialize loss values to zero
        float zero = 0.0f;
        checkCudaError(cudaMemcpy(d_loss, &zero, sizeof(float), cudaMemcpyHostToDevice), "Initializing d_loss");
        checkCudaError(cudaMemcpy(d_loss_optimized, &zero, sizeof(float), cudaMemcpyHostToDevice), "Initializing d_loss_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_X, h_X, data_size, cudaMemcpyHostToDevice), "Copying h_X to d_X");
        checkCudaError(cudaMemcpy(d_Y, h_Y, data_size, cudaMemcpyHostToDevice), "Copying h_Y to d_Y");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            MSE_loss_bf16_origin(d_loss, d_X, d_Y, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };

        auto optimized_kernel = [&]() {
            MSE_loss_bf16_optimized(d_loss_optimized, d_X, d_Y, test_case.N);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(&h_loss_original, d_loss, sizeof(float), cudaMemcpyDeviceToHost), "Copying d_loss to h_loss_original");
        checkCudaError(cudaMemcpy(&h_loss_optimized, d_loss_optimized, sizeof(float), cudaMemcpyDeviceToHost), "Copying d_loss_optimized to h_loss_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        if (!float_equals(h_loss_original, h_loss_optimized, 1e-3f)) {
            printf("Output mismatch: original %.6f, optimized %.6f\n", h_loss_original, h_loss_optimized);
            return 1;
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_X);
        cudaFree(d_Y);
        cudaFree(d_loss);
        cudaFree(d_loss_optimized);

        free(h_X);
        free(h_Y);
        delete [] test_case.X;
        delete [] test_case.Y;
    }

    return 0;
}