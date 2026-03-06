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
__global__ void qkT_bf16_kernel_optimized(int q_seq_len, int kv_seq_len, int dim_qk,
                           const __nv_bfloat16* __restrict__ Q,
                           const __nv_bfloat16* __restrict__ K,
                           __nv_bfloat16* __restrict__ S)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= q_seq_len || col >= kv_seq_len) return;
    
    float acc = 0.0f;
    
    for (int k = 0; k < dim_qk; ++k) {
        float a = __bfloat162float(Q[row * dim_qk + k]);
        float b = __bfloat162float(K[col * dim_qk + k]);
        acc += a * b;
    }
    
    float scale = rsqrtf((float)dim_qk);
    S[row * kv_seq_len + col] = __float2bfloat16(acc * scale);
}

__global__ void softmax_bf16_kernel_optimized(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int q_seq_len,
    int kv_seq_len) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < q_seq_len) {
        const __nv_bfloat16* input_row = input + i * kv_seq_len;
        __nv_bfloat16* out_row = out + i * kv_seq_len;

        float maxval = -FLT_MAX;
        for (int j = 0; j < kv_seq_len; j++) {
            float val = __bfloat162float(input_row[j]);
            if (val > maxval) {
                maxval = val;
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < kv_seq_len; j++) {
            float val = __bfloat162float(input_row[j]);
            float exp_val = expf(val - maxval);
            out_row[j] = __float2bfloat16(exp_val);
            sum += exp_val;
        }
        
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < kv_seq_len; j++) {
            float val = __bfloat162float(out_row[j]);
            out_row[j] = __float2bfloat16(val * inv_sum);
        }
    }
}

__global__ void sv_bf16_kernel_optimized(int q_seq_len, int kv_seq_len, int dim_v,
                          const __nv_bfloat16* __restrict__ S,
                          const __nv_bfloat16* __restrict__ V,
                          __nv_bfloat16* __restrict__ Y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= q_seq_len || col >= dim_v) return;
    
    float acc = 0.0f;
    
    for (int k = 0; k < kv_seq_len; ++k) {
        float a = __bfloat162float(S[row * kv_seq_len + k]);
        float b = __bfloat162float(V[k * dim_v + col]);
        acc += a * b;
    }
    
    Y[row * dim_v + col] =  __float2bfloat16(acc);
}

extern "C" void softmax_attention_bf16_optimized(__nv_bfloat16* Y, const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, 
                                int q_seq_len, int kv_seq_len, int dim_qk, int dim_v)
{
    __nv_bfloat16* S = nullptr;
    __nv_bfloat16* S_softmax = nullptr;
    size_t S_size = (size_t)q_seq_len * (size_t)kv_seq_len * sizeof(__nv_bfloat16);
    cudaMalloc((void**)&S, S_size);
    cudaMalloc((void**)&S_softmax, S_size);

    dim3 block2d(16, 16);

    dim3 grid_qk(
        (kv_seq_len + block2d.x - 1) / block2d.x,
        (q_seq_len + block2d.y - 1) / block2d.y
    );

    qkT_bf16_kernel_optimized<<<grid_qk, block2d>>>(q_seq_len, kv_seq_len, dim_qk, Q, K, S);

    int threadsPerBlock = 256;
    int blocks = (q_seq_len + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax_bf16_kernel_optimized<<<blocks, threadsPerBlock>>>(S_softmax, S, q_seq_len, kv_seq_len);

    dim3 grid_sv(
        (dim_v + block2d.x - 1) / block2d.x,
        (q_seq_len + block2d.y - 1) / block2d.y
    );
    
    sv_bf16_kernel_optimized<<<grid_sv, block2d>>>(q_seq_len, kv_seq_len, dim_v, S_softmax, V, Y);

    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(S_softmax);
}
// ==== OPTIMIZED KERNEL END ====

__global__ void qkT_bf16_kernel(int q_seq_len, int kv_seq_len, int dim_qk,
                           const __nv_bfloat16* __restrict__ Q,
                           const __nv_bfloat16* __restrict__ K,
                           __nv_bfloat16* __restrict__ S)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= q_seq_len || col >= kv_seq_len) return;
    
    float acc = 0.0f;
    
    for (int k = 0; k < dim_qk; ++k) {
        float a = __bfloat162float(Q[row * dim_qk + k]);
        float b = __bfloat162float(K[col * dim_qk + k]);
        acc += a * b;
    }
    
    float scale = rsqrtf((float)dim_qk);
    S[row * kv_seq_len + col] = __float2bfloat16(acc * scale);
}

__global__ void softmax_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    int q_seq_len,
    int kv_seq_len) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < q_seq_len) {
        const __nv_bfloat16* input_row = input + i * kv_seq_len;
        __nv_bfloat16* out_row = out + i * kv_seq_len;

        float maxval = -FLT_MAX;
        for (int j = 0; j < kv_seq_len; j++) {
            float val = __bfloat162float(input_row[j]);
            if (val > maxval) {
                maxval = val;
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < kv_seq_len; j++) {
            float val = __bfloat162float(input_row[j]);
            float exp_val = expf(val - maxval);
            out_row[j] = __float2bfloat16(exp_val);
            sum += exp_val;
        }
        
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < kv_seq_len; j++) {
            float val = __bfloat162float(out_row[j]);
            out_row[j] = __float2bfloat16(val * inv_sum);
        }
    }
}

__global__ void sv_bf16_kernel(int q_seq_len, int kv_seq_len, int dim_v,
                          const __nv_bfloat16* __restrict__ S,
                          const __nv_bfloat16* __restrict__ V,
                          __nv_bfloat16* __restrict__ Y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= q_seq_len || col >= dim_v) return;
    
    float acc = 0.0f;
    
    for (int k = 0; k < kv_seq_len; ++k) {
        float a = __bfloat162float(S[row * kv_seq_len + k]);
        float b = __bfloat162float(V[k * dim_v + col]);
        acc += a * b;
    }
    
    Y[row * dim_v + col] =  __float2bfloat16(acc);
}

extern "C" void softmax_attention_bf16_origin(__nv_bfloat16* Y, const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, 
                                int q_seq_len, int kv_seq_len, int dim_qk, int dim_v)
{
    __nv_bfloat16* S = nullptr;
    __nv_bfloat16* S_softmax = nullptr;
    size_t S_size = (size_t)q_seq_len * (size_t)kv_seq_len * sizeof(__nv_bfloat16);
    cudaMalloc((void**)&S, S_size);
    cudaMalloc((void**)&S_softmax, S_size);

    dim3 block2d(16, 16);

    dim3 grid_qk(
        (kv_seq_len + block2d.x - 1) / block2d.x,
        (q_seq_len + block2d.y - 1) / block2d.y
    );

    qkT_bf16_kernel<<<grid_qk, block2d>>>(q_seq_len, kv_seq_len, dim_qk, Q, K, S);

    int threadsPerBlock = 256;
    int blocks = (q_seq_len + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax_bf16_kernel<<<blocks, threadsPerBlock>>>(S_softmax, S, q_seq_len, kv_seq_len);

    dim3 grid_sv(
        (dim_v + block2d.x - 1) / block2d.x,
        (q_seq_len + block2d.y - 1) / block2d.y
    );
    
    sv_bf16_kernel<<<grid_sv, block2d>>>(q_seq_len, kv_seq_len, dim_v, S_softmax, V, Y);

    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(S_softmax);
}

// Test case input data structure
typedef struct {
    int q_seq_len;
    int kv_seq_len;
    int dim_qk;
    int dim_v;
    __nv_bfloat16 *Q;
    __nv_bfloat16 *K;
    __nv_bfloat16 *V;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<std::tuple<int, int, int, int>> test_configs = {
        {128, 256, 512, 512},
        {256, 512, 1024, 1024},
        {512, 1024, 2048, 2048},
        {1024, 2048, 4096, 4096},
        {256, 1024, 2048, 1024},
        {512, 512, 1024, 2048},
        {1024, 1024, 4096, 2048},
        {128, 2048, 512, 4096}
    };

    for (int i = 0; i < test_configs.size(); i++)
    {
        TestCase test_case;
        test_case.q_seq_len = std::get<0>(test_configs[i]);
        test_case.kv_seq_len = std::get<1>(test_configs[i]);
        test_case.dim_qk = std::get<2>(test_configs[i]);
        test_case.dim_v = std::get<3>(test_configs[i]);

        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        size_t q_size = test_case.q_seq_len * test_case.dim_qk;
        size_t k_size = test_case.kv_seq_len * test_case.dim_qk;
        size_t v_size = test_case.kv_seq_len * test_case.dim_v;
        
        test_case.Q = new __nv_bfloat16[q_size];
        test_case.K = new __nv_bfloat16[k_size];
        test_case.V = new __nv_bfloat16[v_size];
        
        for (size_t ii = 0; ii < q_size; ii++) test_case.Q[ii] = __float2bfloat16(input_dist(rng));
        for (size_t ii = 0; ii < k_size; ii++) test_case.K[ii] = __float2bfloat16(input_dist(rng));
        for (size_t ii = 0; ii < v_size; ii++) test_case.V[ii] = __float2bfloat16(input_dist(rng));
        
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: q_seq_len: %d, kv_seq_len: %d, dim_qk: %d, dim_v: %d. Complexity: %ld\n", test_case.q_seq_len, test_case.kv_seq_len, test_case.dim_qk, test_case.dim_v, (long) test_case.q_seq_len * test_case.kv_seq_len * test_case.dim_qk + test_case.q_seq_len * test_case.kv_seq_len * test_case.dim_v + test_case.q_seq_len * test_case.kv_seq_len);
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
        const int output_item = test_case.q_seq_len * test_case.dim_v;
        size_t q_size = test_case.q_seq_len * test_case.dim_qk * sizeof(__nv_bfloat16);
        size_t k_size = test_case.kv_seq_len * test_case.dim_qk * sizeof(__nv_bfloat16);
        size_t v_size = test_case.kv_seq_len * test_case.dim_v * sizeof(__nv_bfloat16);
        size_t y_size = output_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_Q = (__nv_bfloat16*)malloc(q_size);
        __nv_bfloat16* h_K = (__nv_bfloat16*)malloc(k_size);
        __nv_bfloat16* h_V = (__nv_bfloat16*)malloc(v_size);
        __nv_bfloat16* h_Y = (__nv_bfloat16*)malloc(y_size);
        __nv_bfloat16* h_Y_optimized = (__nv_bfloat16*)malloc(y_size);

        if (!h_Q || !h_K || !h_V || !h_Y || !h_Y_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_Q, test_case.Q, q_size);
        memcpy(h_K, test_case.K, k_size);
        memcpy(h_V, test_case.V, v_size);

        // GPU memory allocation
        __nv_bfloat16 *d_Q, *d_K, *d_V, *d_Y, *d_Y_optimized;
        
        checkCudaError(cudaMalloc((void**)&d_Q, q_size), "Allocating d_Q");
        checkCudaError(cudaMalloc((void**)&d_K, k_size), "Allocating d_K");
        checkCudaError(cudaMalloc((void**)&d_V, v_size), "Allocating d_V");
        checkCudaError(cudaMalloc((void**)&d_Y, y_size), "Allocating d_Y");
        checkCudaError(cudaMalloc((void**)&d_Y_optimized, y_size), "Allocating d_Y_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_Q, h_Q, q_size, cudaMemcpyHostToDevice), "Copying h_Q to d_Q");
        checkCudaError(cudaMemcpy(d_K, h_K, k_size, cudaMemcpyHostToDevice), "Copying h_K to d_K");
        checkCudaError(cudaMemcpy(d_V, h_V, v_size, cudaMemcpyHostToDevice), "Copying h_V to d_V");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */
        
        auto origin_kernel = [&]() {
            softmax_attention_bf16_origin(d_Y, d_Q, d_K, d_V, 
                             test_case.q_seq_len, test_case.kv_seq_len, 
                             test_case.dim_qk, test_case.dim_v);
            checkCudaError(cudaGetLastError(), "Original attention kernel");
        };
        
        auto optimized_kernel = [&]() {
            softmax_attention_bf16_optimized(d_Y_optimized, d_Q, d_K, d_V,
                                       test_case.q_seq_len, test_case.kv_seq_len,
                                       test_case.dim_qk, test_case.dim_v);
            checkCudaError(cudaGetLastError(), "Optimized attention kernel");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_Y, d_Y, y_size, cudaMemcpyDeviceToHost), 
                      "Copying d_Y to h_Y");
        checkCudaError(cudaMemcpy(h_Y_optimized, d_Y_optimized, y_size, cudaMemcpyDeviceToHost), 
                      "Copying d_Y_optimized to h_Y_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < output_item; i++) {
            if (!bfloat16_equals(h_Y[i], h_Y_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_Y[i]), __bfloat162float(h_Y_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_Y);
        cudaFree(d_Y_optimized);
        
        free(h_Q);
        free(h_K);
        free(h_V);
        free(h_Y);
        free(h_Y_optimized);
        delete[] test_case.Q;
        delete[] test_case.K;
        delete[] test_case.V;
    }

    return 0;
}