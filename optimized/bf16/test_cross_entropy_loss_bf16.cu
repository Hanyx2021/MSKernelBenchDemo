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
// Optimized kernel with block-level reduction, loop unrolling, and read-only cache loads
template <unsigned int BLOCK_SIZE>
__global__ void cross_entropy_loss_bf16_kernel_optimized(
    float* loss,
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ Y,
    int C) {
    extern __shared__ float sdata[];

    unsigned int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    const int UNROLL = 4;

    // Unrolled strided loop to increase ILP
    unsigned int i = tid_global;
    unsigned int stride_unroll = stride * UNROLL;
    for (; i + 3u * stride < (unsigned int)C; i += stride_unroll) {
        // Load and accumulate 4 elements
        float x0 = __bfloat162float(__ldg(&X[i]));
        float y0 = __bfloat162float(__ldg(&Y[i]));
        if (y0 != 0.0f) sum += -y0 * logf(fmaxf(x0, 1e-8f));

        unsigned int i1 = i + stride;
        float x1 = __bfloat162float(__ldg(&X[i1]));
        float y1 = __bfloat162float(__ldg(&Y[i1]));
        if (y1 != 0.0f) sum += -y1 * logf(fmaxf(x1, 1e-8f));

        unsigned int i2 = i + 2u * stride;
        float x2 = __bfloat162float(__ldg(&X[i2]));
        float y2 = __bfloat162float(__ldg(&Y[i2]));
        if (y2 != 0.0f) sum += -y2 * logf(fmaxf(x2, 1e-8f));

        unsigned int i3 = i + 3u * stride;
        float x3 = __bfloat162float(__ldg(&X[i3]));
        float y3 = __bfloat162float(__ldg(&Y[i3]));
        if (y3 != 0.0f) sum += -y3 * logf(fmaxf(x3, 1e-8f));
    }

    // Remainder loop
    for (; i < (unsigned int)C; i += stride) {
        float x_val = __bfloat162float(__ldg(&X[i]));
        float y_val = __bfloat162float(__ldg(&Y[i]));
        if (y_val != 0.0f) {
            sum += -y_val * logf(fmaxf(x_val, 1e-8f));
        }
    }

    // Block-level reduction
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // One atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

extern "C" void cross_entropy_loss_bf16_optimized(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int C) {
    const int threadsPerBlock = 256;
    int blocks = (C + threadsPerBlock - 1) / threadsPerBlock;
    blocks = (blocks > 1024 ? 1024 : blocks);
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch optimized kernel
    cross_entropy_loss_bf16_kernel_optimized<threadsPerBlock>
        <<<blocks, threadsPerBlock, sharedMemSize>>>(loss, X, Y, C);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====
 
__global__ void cross_entropy_loss_bf16_kernel(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int C) {

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    
    for (int i = tid; i < C; i += stride) {
        float x_val = __bfloat162float(X[i]);
        float y_val = __bfloat162float(Y[i]);
        
        if (y_val != 0.0f) {
            sum += -1.0 * y_val * logf(fmaxf(x_val, 1e-8f));
        }
    }
    
    if (sum != 0.0f) {
        atomicAdd(loss, sum);
    }
}

extern "C" void cross_entropy_loss_bf16_origin(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int C) {

    const int threadsPerBlock = 256;
    int blocks = (C + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;
    
    cross_entropy_loss_bf16_kernel<<<blocks, threadsPerBlock>>>(loss, X, Y, C);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int C;
    __nv_bfloat16 *X;
    __nv_bfloat16 *Y;
} TestCase;

// Function to load test case from hardcoded values
void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> C_list = {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};

    for (int i = 0; i < C_list.size(); i++) {
        TestCase test_case;
        test_case.C = C_list[i];
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<float> input_dist(0.0f, 1.0f);
        
        int item_count = test_case.C;
        
        // Allocate BF16 memory
        test_case.X = new __nv_bfloat16[item_count];
        test_case.Y = new __nv_bfloat16[item_count];

        float *X_fp32 = (float*)malloc(item_count * sizeof(float));
        float *Y_fp32 = (float*)malloc(item_count * sizeof(float));
        
        // Generate random probabilities and normalize them
        float sum_x = 0.0f, sum_y = 0.0f;
        for (int ii = 0; ii < item_count; ii++) {
            X_fp32[ii] = input_dist(rng);
            sum_x += X_fp32[ii];
            Y_fp32[ii] = input_dist(rng);
            sum_y += Y_fp32[ii];
        }
        
        // Normalize to make them valid probability distributions
        for (int ii = 0; ii < item_count; ii++) {
            X_fp32[ii] /= sum_x;
            Y_fp32[ii] /= sum_y;
        }
        
        for (int i = 0; i < item_count; i++) {
            test_case.X[i] = __float2bfloat16(X_fp32[i]);
            test_case.Y[i] = __float2bfloat16(Y_fp32[i]);
        }
        
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: C: %d. Complexity: %d\n", test_case.C, test_case.C);
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
    
    return total_time / iterations;  // Average time per iteration
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
        const int item_count = test_case.C;
        size_t data_size = item_count * sizeof(__nv_bfloat16);

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
        float *d_loss_original, *d_loss_optimized;
        
        // Allocate BF16 memory
        checkCudaError(cudaMalloc((void**)&d_X, data_size), "Allocating d_X");
        checkCudaError(cudaMalloc((void**)&d_Y, data_size), "Allocating d_Y");
        checkCudaError(cudaMalloc((void**)&d_loss_original, sizeof(float)), "Allocating d_loss_original");
        checkCudaError(cudaMalloc((void**)&d_loss_optimized, sizeof(float)), "Allocating d_loss_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_X, h_X, data_size, cudaMemcpyHostToDevice), "Copying h_X to d_X");
        checkCudaError(cudaMemcpy(d_Y, h_Y, data_size, cudaMemcpyHostToDevice), "Copying h_Y to d_Y");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;

        /* ================  Define test kernels  ================ */
        
        auto origin_kernel = [&]() {
            cudaMemset(d_loss_original, 0, sizeof(float));
            cross_entropy_loss_bf16_origin(d_loss_original, d_X, d_Y, test_case.C);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        auto optimized_kernel = [&]() {
            cudaMemset(d_loss_optimized, 0, sizeof(float));
            cross_entropy_loss_bf16_optimized(d_loss_optimized, d_X, d_Y, test_case.C);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(&h_loss_original, d_loss_original, sizeof(float), cudaMemcpyDeviceToHost), "Copying d_loss_original to host");
        checkCudaError(cudaMemcpy(&h_loss_optimized, d_loss_optimized, sizeof(float), cudaMemcpyDeviceToHost), "Copying d_loss_optimized to host");

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
        cudaFree(d_loss_original);
        cudaFree(d_loss_optimized);

        free(h_X);
        free(h_Y);
        delete [] test_case.X;
        delete [] test_case.Y;
    }

    return 0;
}