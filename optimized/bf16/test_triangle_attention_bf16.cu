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


// Simple function to check if two floats are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====
// Fused one-pass streaming softmax + weighted-sum optimized kernel
__global__ void triangle_attention_bf16_kernel_optimized(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float*         attention_mask,
    __nv_bfloat16*       output,
    int                  batch_size,
    int                  head_num,
    int                  seq_length,
    int                  head_dim
) {
    // Block-per-(b,h,i), thread-per-dimension
    int x = blockIdx.x;
    int d = threadIdx.x;

    int bh_seq = head_num * seq_length;
    int b = x / bh_seq;
    int rem = x % bh_seq;
    int h = rem / seq_length;
    int i = rem % seq_length;

    // Compute flattened index for this (b,h,i,d)
    int idx = ((b * head_num + h) * seq_length + i) * head_dim + d;

    // Load and scale query
    float inv_sqrt_dim = 1.0f / sqrtf((float)head_dim);
    float query_val = __bfloat162float(query[idx]);
    float q_scaled = query_val * inv_sqrt_dim;

    // Base offsets
    int base_k = ((b * head_num + h) * seq_length) * head_dim + d;
    int base_v = base_k;
    int base_m = b * seq_length * seq_length + i * seq_length;

    // Streaming softmax accumulators
    float m = -FLT_MAX;
    float s = 0.0f;
    float z = 0.0f;

    // One-pass over j to compute softmax and weighted sum
    for (int j = 0; j <= i; ++j) {
        int keyIdx  = base_k + j * head_dim;
        int valIdx  = base_v + j * head_dim;
        int maskIdx = base_m + j;

        float k_val    = __bfloat162float(key[keyIdx]);
        float mask_val = attention_mask[maskIdx];
        float curr     = (mask_val > 0.5f) ? -FLT_MAX : q_scaled * k_val;
        float v_val    = __bfloat162float(value[valIdx]);

        if (curr > m) {
            // rescale existing accumulators to new max
            float exp_delta = expf(m - curr);
            z = z * exp_delta + v_val;
            s = s * exp_delta + 1.0f;
            m = curr;
        } else {
            float e = expf(curr - m);
            z += e * v_val;
            s += e;
        }
    }

    // Final output
    float result = (s > 0.0f) ? (z / s) : 0.0f;
    output[idx] = __float2bfloat16(result);
}

extern "C" void triangle_attention_bf16_optimized(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float*         attention_mask,
    __nv_bfloat16*       output,
    int                  batch_size,
    int                  head_num,
    int                  seq_length,
    int                  head_dim
) {
    // Launch configuration: one block per (b, h, i), head_dim threads per block
    int grid_x = batch_size * head_num * seq_length;
    dim3 grid(grid_x, 1, 1);
    dim3 block(head_dim, 1, 1);

    triangle_attention_bf16_kernel_optimized<<<grid, block>>>(
        query, key, value, attention_mask, output,
        batch_size, head_num, seq_length, head_dim
    );
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void triangle_attention_bf16_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float* attention_mask,
    __nv_bfloat16* output,
    int batch_size,
    int head_num,
    int seq_length,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * head_num * seq_length * head_dim;
    
    if (idx >= total_elements) return;

    int b = idx / (head_num * seq_length * head_dim);
    int remaining = idx % (head_num * seq_length * head_dim);
    int h = remaining / (seq_length * head_dim);
    remaining = remaining % (seq_length * head_dim);
    int pos_i = remaining / head_dim;
    int d = remaining % head_dim;
    
    float sum = 0.0f;
    float max_val = -FLT_MAX;

    for (int pos_j = 0; pos_j <= pos_i; pos_j++) {
        float q_val = __bfloat162float(query[idx]);
        int k_idx = b * head_num * seq_length * head_dim + h * seq_length * head_dim + pos_j * head_dim + d;
        float k_val = __bfloat162float(key[k_idx]);
        
        float score = q_val * k_val / sqrtf((float)head_dim);
        
        int mask_idx = b * seq_length * seq_length + pos_i * seq_length + pos_j;
        float mask_val = attention_mask[mask_idx];
        
        if (mask_val > 0.5f) {
            score = -FLT_MAX;
        }
        
        if (score > max_val) max_val = score;
    }
    
    float sum_exp = 0.0f;
    for (int pos_j = 0; pos_j <= pos_i; pos_j++) {
        float q_val = __bfloat162float(query[idx]);
        int k_idx = b * head_num * seq_length * head_dim + h * seq_length * head_dim + pos_j * head_dim + d;
        float k_val = __bfloat162float(key[k_idx]);
        
        float score = q_val * k_val / sqrtf((float)head_dim);
        
        int mask_idx = b * seq_length * seq_length + pos_i * seq_length + pos_j;
        float mask_val = attention_mask[mask_idx];
        
        if (mask_val > 0.5f) {
            score = -FLT_MAX;
        }
        
        float exp_score;
        if (score <= -FLT_MAX) {
            exp_score = 0.0f;
        } else {
            exp_score = expf(score - max_val);
        }
        sum_exp += exp_score;
    }
    
    if (sum_exp > 0.0f) {
        for (int pos_j = 0; pos_j <= pos_i; pos_j++) {
            float q_val = __bfloat162float(query[idx]);
            int k_idx = b * head_num * seq_length * head_dim + h * seq_length * head_dim + pos_j * head_dim + d;
            float k_val = __bfloat162float(key[k_idx]);
            
            float score = q_val * k_val / sqrtf((float)head_dim);
            
            int mask_idx = b * seq_length * seq_length + pos_i * seq_length + pos_j;
            float mask_val = attention_mask[mask_idx];
            
            if (mask_val > 0.5f) {
                score = -FLT_MAX;
            }
            
            float exp_score;
            if (score <= -FLT_MAX) {
                exp_score = 0.0f;
            } else {
                exp_score = expf(score - max_val);
            }
            
            float weight = exp_score / sum_exp;
            
            int v_idx = b * head_num * seq_length * head_dim + h * seq_length * head_dim + pos_j * head_dim + d;
            float v_val = __bfloat162float(value[v_idx]);
            sum += weight * v_val;
        }
    }
    
    output[idx] = __float2bfloat16(sum);
}

extern "C" void triangle_attention_bf16_origin(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float* attention_mask,
    __nv_bfloat16* output,
    int batch_size,
    int head_num,
    int seq_length,
    int head_dim
) {
    int total_elements = batch_size * head_num * seq_length * head_dim;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    triangle_attention_bf16_kernel<<<blocks, threads_per_block>>>(
        query, key, value, attention_mask, output,
        batch_size, head_num, seq_length, head_dim
    );
    
    cudaDeviceSynchronize();
}


// Test case input data structure
typedef struct {
    __nv_bfloat16 *query;
    __nv_bfloat16 *key;
    __nv_bfloat16 *value;
    float *attention_mask;
    int batch_size;
    int head_num;
    int seq_length;
    int head_dim;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<std::tuple<int, int, int, int>> test_configs = {
        {1, 32, 512, 128},
        {1, 96, 256, 128},
        {1, 48, 1024, 128},
        {2, 16, 256, 256},
        {2, 64, 512, 128},
        {4, 8, 32, 128}, 
        {4, 32, 512, 128},
        {8, 16, 512, 128},
    };

    for (int i = 0; i < test_configs.size(); i++)
    {
        TestCase test_case;
        test_case.batch_size = std::get<0>(test_configs[i]);
        test_case.head_num = std::get<1>(test_configs[i]);
        test_case.seq_length = std::get<2>(test_configs[i]);
        test_case.head_dim = std::get<3>(test_configs[i]);

        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        size_t input_size = test_case.batch_size * test_case.head_num * test_case.seq_length * test_case.head_dim;
        size_t mask_size = test_case.batch_size * test_case.seq_length * test_case.seq_length;
        
        test_case.query = new __nv_bfloat16[input_size];
        test_case.key = new __nv_bfloat16[input_size];
        test_case.value = new __nv_bfloat16[input_size];
        test_case.attention_mask = new float[mask_size];
        
        for (size_t ii = 0; ii < input_size; ii++) test_case.query[ii] = __float2bfloat16(input_dist(rng));
        for (size_t ii = 0; ii < input_size; ii++) test_case.key[ii] = __float2bfloat16(input_dist(rng));
        for (size_t ii = 0; ii < input_size; ii++) test_case.value[ii] = __float2bfloat16(input_dist(rng));

        for (size_t ii = 0; ii < mask_size; ii++) {
            test_case.attention_mask[ii] = 0.0f;
        }

        for (int b = 0; b < test_case.batch_size; b++) {
            for (int i = 0; i < test_case.seq_length; i++) {
                for (int j = i + 1; j < test_case.seq_length; j++) {
                    int idx = (b * test_case.seq_length + i) * test_case.seq_length + j;
                    test_case.attention_mask[idx] = 1.0f;
                }
            }
        }

        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: batch_size: %d, head_num: %d, seq_length: %d, head_dim: %d. Complexity: %ld\n", test_case.batch_size,
    test_case.head_num, test_case.seq_length, test_case.head_dim, (long) test_case.batch_size * test_case.head_num * test_case.seq_length * test_case.seq_length * test_case.head_dim);
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
        const int input_item = test_case.batch_size * test_case.head_num * test_case.seq_length * test_case.head_dim;
        const int attention_item = test_case.batch_size * test_case.seq_length * test_case.seq_length;
        size_t q_size = input_item * sizeof(__nv_bfloat16);
        size_t k_size = input_item * sizeof(__nv_bfloat16);
        size_t v_size = input_item * sizeof(__nv_bfloat16);
        size_t o_size = input_item * sizeof(__nv_bfloat16);
        size_t attn_size = attention_item * sizeof(float);

        // Host memory inputs
        __nv_bfloat16* h_Q = (__nv_bfloat16*)malloc(q_size);
        __nv_bfloat16* h_K = (__nv_bfloat16*)malloc(k_size);
        __nv_bfloat16* h_V = (__nv_bfloat16*)malloc(v_size);
        float* h_attn = (float*)malloc(attn_size);
        __nv_bfloat16* h_Y = (__nv_bfloat16*)malloc(o_size);
        __nv_bfloat16* h_Y_optimized = (__nv_bfloat16*)malloc(o_size);

        if (!h_Q || !h_K || !h_V || !h_attn || !h_Y || !h_Y_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_Q, test_case.query, q_size);
        memcpy(h_K, test_case.key, k_size);
        memcpy(h_V, test_case.value, v_size);
        memcpy(h_attn, test_case.attention_mask, attn_size);

        // GPU memory allocation
        __nv_bfloat16 *d_Q, *d_K, *d_V, *d_Y, *d_Y_optimized;
        float *d_attn;
        
        checkCudaError(cudaMalloc((void**)&d_Q, q_size), "Allocating d_Q");
        checkCudaError(cudaMalloc((void**)&d_K, k_size), "Allocating d_K");
        checkCudaError(cudaMalloc((void**)&d_V, v_size), "Allocating d_V");
        checkCudaError(cudaMalloc((void**)&d_attn, attn_size), "Allocating d_attn");
        checkCudaError(cudaMalloc((void**)&d_Y, o_size), "Allocating d_Y");
        checkCudaError(cudaMalloc((void**)&d_Y_optimized, o_size), "Allocating d_Y_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_Q, h_Q, q_size, cudaMemcpyHostToDevice), "Copying h_Q to d_Q");
        checkCudaError(cudaMemcpy(d_K, h_K, k_size, cudaMemcpyHostToDevice), "Copying h_K to d_K");
        checkCudaError(cudaMemcpy(d_V, h_V, v_size, cudaMemcpyHostToDevice), "Copying h_V to d_V");
        checkCudaError(cudaMemcpy(d_attn, h_attn, attn_size, cudaMemcpyHostToDevice), "Copying h_attn to d_attn");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */
        
        auto origin_kernel = [&]() {
            triangle_attention_bf16_origin(d_Q, d_K, d_V, d_attn, d_Y, 
                              test_case.batch_size, test_case.head_num,
                              test_case.seq_length, test_case.head_dim);
            checkCudaError(cudaGetLastError(), "Original attention kernel");
        };
        
        auto optimized_kernel = [&]() {
            triangle_attention_bf16_optimized(d_Q, d_K, d_V, d_attn, d_Y_optimized, 
                              test_case.batch_size, test_case.head_num,
                              test_case.seq_length, test_case.head_dim);
            checkCudaError(cudaGetLastError(), "Optimized attention kernel");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_Y, d_Y, o_size, cudaMemcpyDeviceToHost), 
                      "Copying d_Y to h_Y");
        checkCudaError(cudaMemcpy(h_Y_optimized, d_Y_optimized, o_size, cudaMemcpyDeviceToHost), 
                      "Copying d_Y_optimized to h_Y_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < input_item; i++) {
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
        cudaFree(d_attn);
        cudaFree(d_Y);
        cudaFree(d_Y_optimized);
        
        free(h_Q);
        free(h_K);
        free(h_V);
        free(h_attn);
        free(h_Y);
        free(h_Y_optimized);
        delete[] test_case.query;
        delete[] test_case.key;
        delete[] test_case.value;
        delete[] test_case.attention_mask;
    }

    return 0;
}