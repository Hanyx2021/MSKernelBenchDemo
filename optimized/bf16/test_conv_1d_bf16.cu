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
// Optimized 1D convolution kernel using shared memory for both BF16 kernel weights
// and BF16 input tiles. Dynamic shared memory holds kernel values and input values as floats.
__global__ void conv_1d_bf16_kernel_optimized(const __nv_bfloat16* input,
                                             const __nv_bfloat16* kernel,
                                             __nv_bfloat16* output,
                                             int input_size,
                                             int kernel_size) {
    extern __shared__ float sh_mem[];
    float* s_kernel = sh_mem;  // [0 .. kernel_size-1]
    float* s_input = sh_mem + kernel_size;  // [kernel_size .. kernel_size + tile_size - 1]

    int t = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    int output_size = input_size - kernel_size + 1;

    // Load kernel into shared memory (as float)
    if (t < kernel_size) {
        s_kernel[t] = __bfloat162float(kernel[t]);
    }

    // Cooperative load of input tile: size = blockDim.x + kernel_size - 1
    int tile_size = blockDim.x + kernel_size - 1;
    for (int i = t; i < tile_size; i += blockDim.x) {
        int in_idx = block_start + i;
        if (in_idx < input_size) {
            s_input[i] = __bfloat162float(input[in_idx]);
        } else {
            s_input[i] = 0.0f;
        }
    }
    __syncthreads();

    // Compute convolution for this thread's output element
    int idx = block_start + t;
    if (idx < output_size) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < kernel_size; ++i) {
            acc += s_input[t + i] * s_kernel[i];
        }
        output[idx] = __float2bfloat16(acc);
    }
}

extern "C" void conv_1d_bf16_optimized(const __nv_bfloat16* input,
                                       const __nv_bfloat16* kernel,
                                       __nv_bfloat16* output,
                                       int input_size,
                                       int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    const int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock, 1, 1);
    dim3 grid(blocksPerGrid, 1, 1);

    // Dynamic shared memory: floats for kernel and input tile
    // total floats = kernel_size + (threadsPerBlock + kernel_size - 1)
    size_t shared_mem_bytes = sizeof(float) * (kernel_size + threadsPerBlock + kernel_size - 1);

    conv_1d_bf16_kernel_optimized<<<grid, block, shared_mem_bytes>>>(
        input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void conv_1d_bf16_kernel(const __nv_bfloat16* input, const __nv_bfloat16* kernel, __nv_bfloat16* output,
                                   int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= input_size - kernel_size + 1) return;
    float temp = 0.0f;
    for(int i = 0; i < kernel_size; i++){
        __nv_bfloat16 tempI = input[idx + i];
        __nv_bfloat16 tempK = kernel[i];
        temp += __bfloat162float(tempI) * __bfloat162float(tempK);
    }
    output[idx] = __float2bfloat16(temp);
}

extern "C" void conv_1d_bf16_origin(const __nv_bfloat16* input, const __nv_bfloat16* kernel, __nv_bfloat16* output,
                           int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block(threadsPerBlock, 1, 1);
    dim3 grid(blocksPerGrid, 1, 1);
    
    conv_1d_bf16_kernel<<<grid, block>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int input_size;
    int kernel_size;
    __nv_bfloat16 *input;
    __nv_bfloat16 *kernel;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> input_size_list = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};
    std::vector<int> kernel_size_list = {32, 64, 128, 256};

    for (int i = 0; i < input_size_list.size(); i++)
        for(int j = 0; j < kernel_size_list.size(); j++)
        {
            TestCase test_case;
            test_case.input_size = input_size_list[i];
            test_case.kernel_size = kernel_size_list[j];
            
            // Use fixed seed for reproducibility
            std::random_device rd;
            std::mt19937 rng(rd());  // Random seed for testing
            std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);

            test_case.input = new __nv_bfloat16[test_case.input_size];
            test_case.kernel = new __nv_bfloat16[test_case.kernel_size];
            
            for (int ii = 0; ii < test_case.input_size; ii++) {
                test_case.input[ii] = __float2bfloat16(input_dist(rng));
            }
            for (int ii = 0; ii < test_case.kernel_size; ii++) {
                test_case.kernel[ii] = __float2bfloat16(input_dist(rng));
            }
            test_case_list.push_back(test_case);
        }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: input_size: %d, kernel_size: %d. Complexity: %d\n", test_case.input_size, test_case.kernel_size, test_case.input_size * test_case.kernel_size);
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
        const int input_item = test_case.input_size;
        const int kernel_item = test_case.kernel_size;
        const int output_item = input_item - kernel_item + 1;
        size_t input_size = input_item * sizeof(__nv_bfloat16);
        size_t kernel_size = kernel_item * sizeof(__nv_bfloat16);
        size_t output_size = output_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_input = (__nv_bfloat16*)malloc(input_size);
        __nv_bfloat16* h_kernel = (__nv_bfloat16*)malloc(kernel_size);
        __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(output_size);
        __nv_bfloat16* h_output_optimized = (__nv_bfloat16*)malloc(output_size);

        if (!h_input || !h_kernel || !h_output || !h_output_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_input, test_case.input, input_size);
        memcpy(h_kernel, test_case.kernel, kernel_size);

        // GPU memory allocation
        __nv_bfloat16 *d_input, *d_kernel, *d_output, *d_output_optimized;

        checkCudaError(cudaMalloc((void**)&d_input, input_size), "Allocating d_input");
        checkCudaError(cudaMalloc((void**)&d_kernel, kernel_size), "Allocating d_kernel");
        checkCudaError(cudaMalloc((void**)&d_output, output_size), "Allocating d_output");
        checkCudaError(cudaMalloc((void**)&d_output_optimized, output_size), "Allocating d_output_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "Copying h_input to d_input");
        checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice), "Copying h_kernel to d_kernel");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            conv_1d_bf16_origin(d_input, d_kernel, d_output, test_case.input_size, test_case.kernel_size);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            conv_1d_bf16_optimized(d_input, d_kernel, d_output_optimized, test_case.input_size, test_case.kernel_size);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "Copying d_output to host");
        checkCudaError(cudaMemcpy(h_output_optimized, d_output_optimized, output_size, cudaMemcpyDeviceToHost), "Copying d_output_optimized to host");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < output_item; i++) {
            if (!bfloat16_equals(h_output[i], h_output_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_output[i]), __bfloat162float(h_output_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);
        cudaFree(d_output_optimized);

        free(h_input);
        free(h_kernel);
        free(h_output);
        free(h_output_optimized);
        delete [] test_case.input;
        delete [] test_case.kernel;
    }

    return 0;
}