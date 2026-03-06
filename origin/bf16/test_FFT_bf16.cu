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
bool bfloat16_equals_relative(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    float fa = __bfloat162float(a);
    float fb = __bfloat162float(b);
    
    if (fa == fb) return true;
    if (fa == 0.0f || fb == 0.0f) {
        return fabs(fa - fb) < tolerance;
    }
    return fabs(fa - fb) / fmax(fabs(fa), fabs(fb)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====

__global__ void bit_reverse_permute_bf16_kernel_optimized(
    __nv_bfloat16* out_real, __nv_bfloat16* out_img,
    const __nv_bfloat16* in_real, const __nv_bfloat16* in_img,
    int N, const int* bit_rev) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    int rev_idx = bit_rev[idx];
    out_real[idx] = in_real[rev_idx];
    out_img[idx] = in_img[rev_idx];
}

__global__ void butterfly_bf16_kernel_optimized(
    __nv_bfloat16* data_real, __nv_bfloat16* data_img,
    const __nv_bfloat16* twiddle_real, const __nv_bfloat16* twiddle_img,
    int N, int stage) {
    
    int butterfly_size = 1 << stage;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int num_butterflies = N >> 1;
    
    if (idx >= num_butterflies) return;
    
    int group_id = idx / (butterfly_size >> 1);
    int pair_id = idx % (butterfly_size >> 1);
    
    int idx1 = group_id * butterfly_size + pair_id;
    int idx2 = idx1 + (butterfly_size >> 1);
    
    float x1_real = __bfloat162float(data_real[idx1]);
    float x1_img = __bfloat162float(data_img[idx1]);
    float x2_real = __bfloat162float(data_real[idx2]);
    float x2_img = __bfloat162float(data_img[idx2]);
    
    int twiddle_idx = pair_id * (N >> stage);
    
    float tw_real = __bfloat162float(twiddle_real[twiddle_idx]);
    float tw_img = __bfloat162float(twiddle_img[twiddle_idx]);
    
    float y_real = tw_real * x2_real - tw_img * x2_img;
    float y_img = tw_real * x2_img + tw_img * x2_real;
    
    data_real[idx1] = __float2bfloat16(x1_real + y_real);
    data_img[idx1] = __float2bfloat16(x1_img + y_img);
    data_real[idx2] = __float2bfloat16(x1_real - y_real);
    data_img[idx2] = __float2bfloat16(x1_img - y_img);
}

int* compute_bit_reversal_table_optimized(int N) {
    int* table = new int[N];
    int logN = 0;
    for (int i = 1; i < N; i <<= 1)
        logN++;
    
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int j = 0; j < logN; j++) {
            rev = (rev << 1) | ((i >> j) & 1);
        }
        table[i] = rev;
    }
    return table;
}

void compute_twiddle_factors_bf16_optimized(__nv_bfloat16* real, __nv_bfloat16* img, int N) {
    for (int i = 0; i < N/2; i++) {
        float angle = -2.0f * M_PI * i / N;
        double a = angle;
        real[i] = __float2bfloat16(cos(a));
        img[i] = __float2bfloat16(sin(a));
    }
}

extern "C" void FFT_bf16_optimized(
    const __nv_bfloat16* input_real, 
    const __nv_bfloat16* input_img, 
    __nv_bfloat16* output_real, 
    __nv_bfloat16* output_img, 
    int N) 
{
    __nv_bfloat16 *d_input_real, *d_input_img;
    __nv_bfloat16 *d_work_real, *d_work_img;
    __nv_bfloat16 *d_twiddle_real, *d_twiddle_img;
    int *d_bit_rev;
    
    size_t mem_size = N * sizeof(__nv_bfloat16);
    
    cudaMalloc(&d_input_real, mem_size);
    cudaMalloc(&d_input_img, mem_size);
    cudaMalloc(&d_work_real, mem_size);
    cudaMalloc(&d_work_img, mem_size);
    
    cudaMemcpy(d_input_real, input_real, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_img, input_img, mem_size, cudaMemcpyHostToDevice);
    
    int* h_bit_rev = compute_bit_reversal_table_optimized(N);
    cudaMalloc(&d_bit_rev, N * sizeof(int));
    cudaMemcpy(d_bit_rev, h_bit_rev, N * sizeof(int), cudaMemcpyHostToDevice);
    
    __nv_bfloat16* h_twiddle_real = new __nv_bfloat16[N/2];
    __nv_bfloat16* h_twiddle_img = new __nv_bfloat16[N/2];
    compute_twiddle_factors_bf16_optimized(h_twiddle_real, h_twiddle_img, N);
    
    cudaMalloc(&d_twiddle_real, (N/2) * sizeof(__nv_bfloat16));
    cudaMalloc(&d_twiddle_img, (N/2) * sizeof(__nv_bfloat16));
    cudaMemcpy(d_twiddle_real, h_twiddle_real, (N/2) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddle_img, h_twiddle_img, (N/2) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    bit_reverse_permute_bf16_kernel_optimized<<<blocks, threads_per_block>>>(
        d_work_real, d_work_img, 
        d_input_real, d_input_img, 
        N, d_bit_rev);
    
    cudaDeviceSynchronize();
    
    int stages = 0;
    for (int n = N; n > 1; n >>= 1)
        stages++;
    
    for (int stage = 1; stage <= stages; stage++) {
        int butterflies = N >> 1;
        blocks = (butterflies + threads_per_block - 1) / threads_per_block;
        
        butterfly_bf16_kernel_optimized<<<blocks, threads_per_block>>>(
            d_work_real, d_work_img,
            d_twiddle_real, d_twiddle_img,
            N, stage);
        
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(output_real, d_work_real, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_img, d_work_img, mem_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input_real);
    cudaFree(d_input_img);
    cudaFree(d_work_real);
    cudaFree(d_work_img);
    cudaFree(d_bit_rev);
    cudaFree(d_twiddle_real);
    cudaFree(d_twiddle_img);
    delete[] h_bit_rev;
    delete[] h_twiddle_real;
    delete[] h_twiddle_img;
}

// ==== OPTIMIZED KERNEL END ====

__global__ void bit_reverse_permute_bf16_kernel(
    __nv_bfloat16* out_real, __nv_bfloat16* out_img,
    const __nv_bfloat16* in_real, const __nv_bfloat16* in_img,
    int N, const int* bit_rev) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    int rev_idx = bit_rev[idx];
    out_real[idx] = in_real[rev_idx];
    out_img[idx] = in_img[rev_idx];
}

__global__ void butterfly_bf16_kernel(
    __nv_bfloat16* data_real, __nv_bfloat16* data_img,
    const __nv_bfloat16* twiddle_real, const __nv_bfloat16* twiddle_img,
    int N, int stage) {
    
    int butterfly_size = 1 << stage;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int num_butterflies = N >> 1;
    
    if (idx >= num_butterflies) return;
    
    int group_id = idx / (butterfly_size >> 1);
    int pair_id = idx % (butterfly_size >> 1);
    
    int idx1 = group_id * butterfly_size + pair_id;
    int idx2 = idx1 + (butterfly_size >> 1);
    
    float x1_real = __bfloat162float(data_real[idx1]);
    float x1_img = __bfloat162float(data_img[idx1]);
    float x2_real = __bfloat162float(data_real[idx2]);
    float x2_img = __bfloat162float(data_img[idx2]);
    
    int twiddle_idx = pair_id * (N >> stage);
    
    float tw_real = __bfloat162float(twiddle_real[twiddle_idx]);
    float tw_img = __bfloat162float(twiddle_img[twiddle_idx]);
    
    float y_real = tw_real * x2_real - tw_img * x2_img;
    float y_img = tw_real * x2_img + tw_img * x2_real;
    
    data_real[idx1] = __float2bfloat16(x1_real + y_real);
    data_img[idx1] = __float2bfloat16(x1_img + y_img);
    data_real[idx2] = __float2bfloat16(x1_real - y_real);
    data_img[idx2] = __float2bfloat16(x1_img - y_img);
}

int* compute_bit_reversal_table_origin(int N) {
    int* table = new int[N];
    int logN = 0;
    for (int i = 1; i < N; i <<= 1)
        logN++;
    
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int j = 0; j < logN; j++) {
            rev = (rev << 1) | ((i >> j) & 1);
        }
        table[i] = rev;
    }
    return table;
}

void compute_twiddle_factors_bf16(__nv_bfloat16* real, __nv_bfloat16* img, int N) {
    for (int i = 0; i < N/2; i++) {
        float angle = -2.0f * M_PI * i / N;
        double a = angle;
        real[i] = __float2bfloat16(cos(a));
        img[i] = __float2bfloat16(sin(a));
    }
}

extern "C" void FFT_bf16_origin(
    const __nv_bfloat16* input_real, 
    const __nv_bfloat16* input_img, 
    __nv_bfloat16* output_real, 
    __nv_bfloat16* output_img, 
    int N) 
{
    __nv_bfloat16 *d_input_real, *d_input_img;
    __nv_bfloat16 *d_work_real, *d_work_img;
    __nv_bfloat16 *d_twiddle_real, *d_twiddle_img;
    int *d_bit_rev;
    
    size_t mem_size = N * sizeof(__nv_bfloat16);
    
    cudaMalloc(&d_input_real, mem_size);
    cudaMalloc(&d_input_img, mem_size);
    cudaMalloc(&d_work_real, mem_size);
    cudaMalloc(&d_work_img, mem_size);
    
    cudaMemcpy(d_input_real, input_real, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_img, input_img, mem_size, cudaMemcpyHostToDevice);
    
    int* h_bit_rev = compute_bit_reversal_table_origin(N);
    cudaMalloc(&d_bit_rev, N * sizeof(int));
    cudaMemcpy(d_bit_rev, h_bit_rev, N * sizeof(int), cudaMemcpyHostToDevice);
    
    __nv_bfloat16* h_twiddle_real = new __nv_bfloat16[N/2];
    __nv_bfloat16* h_twiddle_img = new __nv_bfloat16[N/2];
    compute_twiddle_factors_bf16(h_twiddle_real, h_twiddle_img, N);
    
    cudaMalloc(&d_twiddle_real, (N/2) * sizeof(__nv_bfloat16));
    cudaMalloc(&d_twiddle_img, (N/2) * sizeof(__nv_bfloat16));
    cudaMemcpy(d_twiddle_real, h_twiddle_real, (N/2) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddle_img, h_twiddle_img, (N/2) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    bit_reverse_permute_bf16_kernel<<<blocks, threads_per_block>>>(
        d_work_real, d_work_img, 
        d_input_real, d_input_img, 
        N, d_bit_rev);
    
    cudaDeviceSynchronize();
    
    int stages = 0;
    for (int n = N; n > 1; n >>= 1)
        stages++;
    
    for (int stage = 1; stage <= stages; stage++) {
        int butterflies = N >> 1;
        blocks = (butterflies + threads_per_block - 1) / threads_per_block;
        
        butterfly_bf16_kernel<<<blocks, threads_per_block>>>(
            d_work_real, d_work_img,
            d_twiddle_real, d_twiddle_img,
            N, stage);
        
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(output_real, d_work_real, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_img, d_work_img, mem_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input_real);
    cudaFree(d_input_img);
    cudaFree(d_work_real);
    cudaFree(d_work_img);
    cudaFree(d_bit_rev);
    cudaFree(d_twiddle_real);
    cudaFree(d_twiddle_img);
    delete[] h_bit_rev;
    delete[] h_twiddle_real;
    delete[] h_twiddle_img;
}


// Test case input data structure
typedef struct {
    int N;
    __nv_bfloat16 *input_real;
    __nv_bfloat16 *input_img;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14};

    for (int i = 0; i < N_list.size(); i++) {
        TestCase test_case;
        test_case.N = N_list[i];
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int input_item = test_case.N;
        test_case.input_real = new __nv_bfloat16[input_item];
        test_case.input_img = new __nv_bfloat16[input_item];
        
        for (int ii = 0; ii < input_item; ii++) {
            test_case.input_real[ii] = __float2bfloat16(input_dist(rng));
            test_case.input_img[ii] = __float2bfloat16(input_dist(rng));
        }
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: N: %d. Complexity: %d\n", test_case.N, (int)(test_case.N * log2((double)test_case.N)));
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
        size_t input_size = input_item * sizeof(__nv_bfloat16);
        size_t output_size = input_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_input_real = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_input_img = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_output_real = (__nv_bfloat16*)malloc(output_size);
        __nv_bfloat16* h_output_img = (__nv_bfloat16*)malloc(output_size);
        __nv_bfloat16* h_output_real_optimized = (__nv_bfloat16*)malloc(output_size);
        __nv_bfloat16* h_output_img_optimized = (__nv_bfloat16*)malloc(output_size);

        if (!h_input_real || !h_input_img || !h_output_real || !h_output_img || !h_output_real_optimized || !h_output_img_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_input_real, test_case.input_real, input_size);
        memcpy(h_input_img, test_case.input_img, input_size);

        // GPU memory allocation
        __nv_bfloat16 *d_input_real, *d_input_img, *d_output_real, *d_output_img, *d_output_real_optimized, *d_output_img_optimized;

        checkCudaError(cudaMalloc((void**)&d_input_real, input_size), "Allocating d_input_real");
        checkCudaError(cudaMalloc((void**)&d_input_img, input_size), "Allocating d_input_img");
        checkCudaError(cudaMalloc((void**)&d_output_real, output_size), "Allocating d_output_real");
        checkCudaError(cudaMalloc((void**)&d_output_img, output_size), "Allocating d_output_img");
        checkCudaError(cudaMalloc((void**)&d_output_real_optimized, output_size), "Allocating d_output_real_optimized");
        checkCudaError(cudaMalloc((void**)&d_output_img_optimized, output_size), "Allocating d_output_img_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_input_real, h_input_real, input_size, cudaMemcpyHostToDevice), "Copying h_input_real to d_input_real");
        checkCudaError(cudaMemcpy(d_input_img, h_input_img, input_size, cudaMemcpyHostToDevice), "Copying h_input_img to d_input_img");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            FFT_bf16_origin(d_input_real, d_input_img, d_output_real, d_output_img, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            FFT_bf16_optimized(d_input_real, d_input_img, d_output_real_optimized, d_output_img_optimized, test_case.N);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_output_real, d_output_real, output_size, cudaMemcpyDeviceToHost), "Copying d_output_real to h_output_real");
        checkCudaError(cudaMemcpy(h_output_img, d_output_img, output_size, cudaMemcpyDeviceToHost), "Copying d_output_img to h_output_img");
        checkCudaError(cudaMemcpy(h_output_real_optimized, d_output_real_optimized, output_size, cudaMemcpyDeviceToHost), "Copying d_output_real_optimized to h_output_real_optimized");
        checkCudaError(cudaMemcpy(h_output_img_optimized, d_output_img_optimized, output_size, cudaMemcpyDeviceToHost), "Copying d_output_img_optimized to h_output_img_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        int error_num = 0;

        for (int i = 0; i < input_item; i++) {
            if (!bfloat16_equals_relative(h_output_real[i], h_output_real_optimized[i], 1e-3f)) {
                error_num += 1;
                if(error_num >= test_case.N * 0.01) {
                    printf("Output Real mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_output_real[i]), __bfloat162float(h_output_real_optimized[i]));
                    return 1;
                }
            }
            if (!bfloat16_equals_relative(h_output_img[i], h_output_img_optimized[i], 1e-3f)) {
                error_num += 1;
                if(error_num >= test_case.N * 0.01) {
                    printf("Output Img mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_output_img[i]), __bfloat162float(h_output_img_optimized[i]));
                    return 1;
                }
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_input_real);
        cudaFree(d_input_img);
        cudaFree(d_output_real);
        cudaFree(d_output_img);
        cudaFree(d_output_real_optimized);
        cudaFree(d_output_img_optimized);

        free(h_input_real);
        free(h_input_img);
        free(h_output_real);
        free(h_output_img);
        free(h_output_real_optimized);
        free(h_output_img_optimized);
        delete [] test_case.input_real;
        delete [] test_case.input_img;
    }

    return 0;
}