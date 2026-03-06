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
// Maximum D/2 supported in constant memory (adjust if needed)
#define MAX_D2 8192

// Precomputed theta values in constant memory
__constant__ float theta_const[MAX_D2];

// Configuration structure for optimized launch parameters
struct RopeBf16OptimizedConfig {
    dim3 grid;
    dim3 block;
};

// Optimized kernel with fused sincos and 32-bit vectorized loads/stores
__global__ void rope_bf16_kernel_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int M,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d2 = D / 2;
    int total_pairs = M * d2;
    if (idx >= total_pairs) return;

    int m = idx / d2;
    int pair = idx % d2;

    // Load two bfloat16s at once via 32-bit transaction
    const __nv_bfloat162* input32 = reinterpret_cast<const __nv_bfloat162*>(input);
    __nv_bfloat162 packed_in = __ldg(&input32[idx]);

    // De-pack manually using scalar conversion
    float q0 = __bfloat162float(packed_in.x);
    float q1 = __bfloat162float(packed_in.y);

    // Lookup precomputed theta via read-only cache
    float theta = __ldg(&theta_const[pair]);
    float m_theta = static_cast<float>(m) * theta;

    // Compute sin and cos via fused intrinsic
    float sinv, cosv;
    sincosf(m_theta, &sinv, &cosv);

    // Apply rotary transform
    float2 out;
    out.x = q0 * cosv - q1 * sinv;
    out.y = q0 * sinv + q1 * cosv;

    // Pack and store results in one 32-bit transaction
    __nv_bfloat162 packed_out;
    packed_out.x = __float2bfloat16(out.x);
    packed_out.y = __float2bfloat16(out.y);

    __nv_bfloat162* output32 = reinterpret_cast<__nv_bfloat162*>(output);
    output32[idx] = packed_out;
}

// External C function wrapper
extern "C" void rope_bf16_optimized(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int M,
    int D,
    float base
) {
    int d2 = D / 2;
    // Precompute theta on host
    std::vector<float> theta_host(d2);
    for (int k = 0; k < d2; ++k) {
        theta_host[k] = powf(base, -2.0f * static_cast<float>(k) / static_cast<float>(D));
    }
    // Copy to constant memory
    cudaMemcpyToSymbol(theta_const, theta_host.data(), d2 * sizeof(float));

    // Launch kernel with optimized vectorized load/store
    int total_pairs = M * d2;
    const int threads = 512;
    int blocks = (total_pairs + threads - 1) / threads;

    RopeBf16OptimizedConfig config;
    config.block = dim3(threads, 1, 1);
    config.grid  = dim3(blocks,  1, 1);

    rope_bf16_kernel_optimized<<<config.grid, config.block>>>(output, input, M, D);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void rope_bf16_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int M,
    const int D,
    const float base = 10000.0f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = M * (D / 2);
    if (idx >= total_pairs) return;

    int m = idx / (D / 2);
    int pair = idx % (D / 2);
    int i = pair * 2;
    int data_offset = m * D + i;

    float q0 = __bfloat162float(input[data_offset]);
    float q1 = __bfloat162float(input[data_offset + 1]);

    float theta = powf(base, -2.0f * (float)pair / (float)D);
    float m_theta = (float)m * theta;

    float cos_val = cosf(m_theta);
    float sin_val = sinf(m_theta);

    float out0 = q0 * cos_val - q1 * sin_val;
    float out1 = q0 * sin_val + q1 * cos_val;

    output[data_offset] = __float2bfloat16(out0);
    output[data_offset + 1] = __float2bfloat16(out1);
}

extern "C" void rope_bf16_origin(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int M,
    int D,
    float base
) {
    int total_pairs = M * (D / 2);
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    rope_bf16_kernel<<<blocks, threads>>>(output, input, M, D, base);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int M;
    int D;
    __nv_bfloat16 *input;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> M_list = {16, 32, 64, 128};
    std::vector<int> D_list = {1 << 8, 1 << 10, 1 << 12, 1 << 14};

    for (int i = 0; i < M_list.size(); i++) 
        for(int j = 0 ; j < D_list.size(); j++)
        {
            TestCase test_case;
            test_case.M = M_list[i];
            test_case.D = D_list[j];

            // Use fixed seed for reproducibility
            std::random_device rd;
            std::mt19937 rng(rd());  // Random seed for testing
            std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
            
            int input_item = test_case.M * test_case.D;
            test_case.input = new __nv_bfloat16[input_item];
            
            for (int ii = 0; ii < input_item; ii++) {
                test_case.input[ii] = __float2bfloat16(input_dist(rng));
            }
            test_case_list.push_back(test_case);
        }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: M: %d, D: %d. Complexity: %d\n", test_case.M, test_case.D, test_case.M * test_case.D);
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
        const int input_item = test_case.M * test_case.D;
        size_t input_size = input_item * sizeof(__nv_bfloat16);
        size_t output_size = input_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_input = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(output_size);
        __nv_bfloat16* h_output_optimized = (__nv_bfloat16*)malloc(output_size);

        if (!h_input || !h_output || !h_output_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_input, test_case.input, input_size);

        // GPU memory allocation
        __nv_bfloat16 *d_input, *d_output, *d_output_optimized;

        checkCudaError(cudaMalloc((void**)&d_input, input_size), "Allocating d_input");
        checkCudaError(cudaMalloc((void**)&d_output, output_size), "Allocating d_output");
        checkCudaError(cudaMalloc((void**)&d_output_optimized, output_size), "Allocating d_output_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "Copying h_input to d_input");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */
        
        auto origin_kernel = [&]() {
            rope_bf16_origin(d_output, d_input, test_case.M, test_case.D, 10000.0f);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            rope_bf16_optimized(d_output_optimized, d_input, test_case.M, test_case.D, 10000.0f);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "Copying d_output to h_output");
        checkCudaError(cudaMemcpy(h_output_optimized, d_output_optimized, output_size, cudaMemcpyDeviceToHost), "Copying d_output_optimized to h_output_optimized");

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
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_output_optimized);

        free(h_input);
        free(h_output);
        free(h_output_optimized);
        delete [] test_case.input;
    }

    return 0;
}