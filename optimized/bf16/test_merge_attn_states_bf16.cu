#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <functional>
#include <math.h>
#include <optional>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>


// Simple function to check if two bf16 values are approximately equal
bool bfloat16_equals(__nv_bfloat16 a, __nv_bfloat16 b, float tolerance) {
    return fabs(__bfloat162float(a) - __bfloat162float(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====
// Optimized kernel using read-only data cache and __restrict__ pointers
__global__ void merge_attn_states_bf16_kernel_optimized(
    __nv_bfloat162* __restrict__ V_out,
    __nv_bfloat16* __restrict__ LSE_out,
    const __nv_bfloat162* __restrict__ V_a,
    const __nv_bfloat16* __restrict__ lse_a,
    const __nv_bfloat162* __restrict__ V_b,
    const __nv_bfloat16* __restrict__ lse_b,
    int num_tokens,
    int num_heads,
    int head_size) {
    int head_idx = blockIdx.y;
    int token_idx = blockIdx.x;
    int pair_idx = threadIdx.x;

    int head_pairs = (head_size + 1) / 2;
    if (head_idx >= num_heads || token_idx >= num_tokens || pair_idx >= head_pairs) return;

    __shared__ float p_scale, s_scale, new_lse;
    int lse_index = head_idx * num_tokens + token_idx;
    if (pair_idx == 0) {
        // Load lse values through read-only cache
        float p_lse = __bfloat162float(__ldg(&lse_a[lse_index]));
        float s_lse = __bfloat162float(__ldg(&lse_b[lse_index]));
        // handle infinities
        if (isinf(p_lse)) p_lse = -FLT_MAX;
        if (isinf(s_lse)) s_lse = -FLT_MAX;
        float max_lse = fmaxf(p_lse, s_lse);
        float p_exp = expf(p_lse - max_lse);
        float s_exp = expf(s_lse - max_lse);
        float total = p_exp + s_exp;
        p_scale = p_exp / total;
        s_scale = s_exp / total;
        new_lse = logf(total) + max_lse;
    }
    __syncthreads();

    // Compute offset in vectorized array
    int pair_offset = token_idx * (num_heads * head_pairs)
                      + head_idx * head_pairs + pair_idx;

    // Load bf16 pairs from inputs via read-only cache
    __nv_bfloat162 va_pair = __ldg(&V_a[pair_offset]);
    __nv_bfloat162 vb_pair = __ldg(&V_b[pair_offset]);

    // Unpack to floats and compute weighted sum
    float va0 = __bfloat162float(va_pair.x);
    float vb0 = __bfloat162float(vb_pair.x);
    float r0 = va0 * p_scale + vb0 * s_scale;
    __nv_bfloat16 out0 = __float2bfloat16(r0);

    // Check if second element exists (odd head_size guard)
    bool has_second = (pair_idx * 2 + 1 < head_size);
    __nv_bfloat16 out1;
    if (has_second) {
        float va1 = __bfloat162float(va_pair.y);
        float vb1 = __bfloat162float(vb_pair.y);
        float r1 = va1 * p_scale + vb1 * s_scale;
        out1 = __float2bfloat16(r1);
    } else {
        out1 = va_pair.y;  // padding for odd element
    }

    // Pack and store result
    __nv_bfloat162 out_pair;
    out_pair.x = out0;
    out_pair.y = out1;
    V_out[pair_offset] = out_pair;

    // Write new LSE once per token-head
    if (LSE_out != nullptr && pair_idx == 0) {
        LSE_out[lse_index] = __float2bfloat16(new_lse);
    }
}

// External C wrapper with optimized kernel invocation
extern "C" void merge_attn_states_bf16_optimized(
    __nv_bfloat16* __restrict__ V_out,
    __nv_bfloat16* __restrict__ LSE_out,
    const __nv_bfloat16* __restrict__ V_a,
    const __nv_bfloat16* __restrict__ lse_a,
    const __nv_bfloat16* __restrict__ V_b,
    const __nv_bfloat16* __restrict__ lse_b,
    int num_tokens,
    int num_heads,
    int head_size) {
    int head_pairs = (head_size + 1) / 2;
    dim3 grid(num_tokens, num_heads);
    dim3 block(head_pairs);

    // Reinterpret raw bf16 pointers as bf16-pair pointers
    auto V_a_pair = reinterpret_cast<const __nv_bfloat162* __restrict__>(V_a);
    auto V_b_pair = reinterpret_cast<const __nv_bfloat162* __restrict__>(V_b);
    auto V_out_pair = reinterpret_cast<__nv_bfloat162* __restrict__>(V_out);

    merge_attn_states_bf16_kernel_optimized<<<grid, block>>>(
        V_out_pair, LSE_out,
        V_a_pair, lse_a,
        V_b_pair, lse_b,
        num_tokens, num_heads, head_size);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void merge_attn_states_bf16_kernel(
    __nv_bfloat16* V_out, __nv_bfloat16* LSE_out, const __nv_bfloat16* V_a,
    const __nv_bfloat16* lse_a, const __nv_bfloat16* V_b,
    const __nv_bfloat16* lse_b, const int num_tokens, const int num_heads,
    const int head_size) {
    
    const int head_idx = blockIdx.y;
    const int token_idx = blockIdx.x;
    const int element_idx = threadIdx.x;
    
    if (head_idx >= num_heads || token_idx >= num_tokens || element_idx >= head_size) return;
    
    __shared__ float p_scale;
    __shared__ float s_scale;
    __shared__ float new_lse;
    
    if (threadIdx.x == 0) {
        float p_lse = __bfloat162float(lse_a[head_idx * num_tokens + token_idx]);
        float s_lse = __bfloat162float(lse_b[head_idx * num_tokens + token_idx]);
        
        p_lse = isinf(p_lse) ? -FLT_MAX : p_lse;
        s_lse = isinf(s_lse) ? -FLT_MAX : s_lse;
        
        const float max_lse = fmaxf(p_lse, s_lse);
        const float p_exp = expf(p_lse - max_lse);
        const float s_exp = expf(s_lse - max_lse);
        const float total_exp = p_exp + s_exp;
        
        p_scale = p_exp / total_exp;
        s_scale = s_exp / total_exp;
        new_lse = logf(total_exp) + max_lse;
    }
    
    __syncthreads();
    
    const int src_offset = token_idx * num_heads * head_size + 
                          head_idx * head_size + element_idx;
    
    float v_a_f = __bfloat162float(V_a[src_offset]);
    float v_b_f = __bfloat162float(V_b[src_offset]);
    
    float result_f = v_a_f * p_scale + v_b_f * s_scale;
    
    V_out[src_offset] = __float2bfloat16(result_f);
    
    if (LSE_out != nullptr && element_idx == 0) {
        LSE_out[head_idx * num_tokens + token_idx] = __float2bfloat16(new_lse);
    }
}

extern "C" void merge_attn_states_bf16_origin(
    __nv_bfloat16* V_out, 
    __nv_bfloat16* LSE_out, 
    const __nv_bfloat16* V_a,
    const __nv_bfloat16* lse_a, 
    const __nv_bfloat16* V_b,
    const __nv_bfloat16* lse_b, 
    int num_tokens, 
    int num_heads,
    int head_size) {
    
    dim3 grid(num_tokens, num_heads, 1);
    dim3 block(head_size, 1, 1);
    
    merge_attn_states_bf16_kernel<<<grid, block>>>(
        V_out, LSE_out, V_a, lse_a, V_b, lse_b, num_tokens, num_heads, head_size);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int num_tokens;
    int num_heads;
    int head_size;
    __nv_bfloat16 *V_a;
    __nv_bfloat16 *V_b;
    __nv_bfloat16 *LSE_a;
    __nv_bfloat16 *LSE_b;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> num_token_list = {512, 768, 1024};
    std::vector<int> num_head_list = {32, 64};
    std::vector<int> head_size_list = {128, 256};

    for (int i = 0; i < num_token_list.size(); i++) 
        for (int j = 0; j < num_head_list.size(); j++)
            for(int k = 0; k < head_size_list.size(); k++)
            {
                TestCase test_case;
                test_case.num_tokens = num_token_list[i];
                test_case.num_heads = num_head_list[j];
                test_case.head_size = head_size_list[k];
                
                // Use fixed seed for reproducibility
                std::random_device rd;
                std::mt19937 rng(rd());  // Random seed for testing
                std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
                
                int V_item = test_case.num_tokens * test_case.num_heads * test_case.head_size;
                int LSE_item = test_case.num_tokens * test_case.num_heads;
                test_case.V_a = new __nv_bfloat16[V_item];
                test_case.V_b = new __nv_bfloat16[V_item];
                test_case.LSE_a = new __nv_bfloat16[LSE_item];
                test_case.LSE_b = new __nv_bfloat16[LSE_item];
                
                for (int ii = 0; ii < V_item; ii++) {
                    test_case.V_a[ii] = __float2bfloat16(input_dist(rng));
                    test_case.V_b[ii] = __float2bfloat16(input_dist(rng));
                }
                for (int jj = 0; jj < LSE_item; jj++) {
                    test_case.LSE_a[jj] = __float2bfloat16(input_dist(rng));
                    test_case.LSE_b[jj] = __float2bfloat16(input_dist(rng));
                }
                test_case_list.push_back(test_case);
            }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: num_tokens: %d, num_heads: %d, head_size: %d. Complexity: %d\n", 
        test_case.num_tokens, test_case.num_heads, test_case.head_size, test_case.num_tokens * test_case.num_heads * test_case.head_size);
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
        const int num_tokens = test_case.num_tokens;
        const int num_heads = test_case.num_heads;
        const int head_size = test_case.head_size;
        int v_item = num_tokens * num_heads * head_size;
        size_t v_size = v_item * sizeof(__nv_bfloat16);
        int lse_item = num_tokens * num_heads;
        size_t lse_size = lse_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_v_a = (__nv_bfloat16*)malloc(v_size);
        __nv_bfloat16* h_v_b = (__nv_bfloat16*)malloc(v_size);
        __nv_bfloat16* h_v_output = (__nv_bfloat16*)malloc(v_size);
        __nv_bfloat16* h_v_output_optimized = (__nv_bfloat16*)malloc(v_size);

        __nv_bfloat16* h_lse_a = (__nv_bfloat16*)malloc(lse_size);
        __nv_bfloat16* h_lse_b = (__nv_bfloat16*)malloc(lse_size);
        __nv_bfloat16* h_lse_output = (__nv_bfloat16*)malloc(lse_size);
        __nv_bfloat16* h_lse_output_optimized = (__nv_bfloat16*)malloc(lse_size);

        if (!h_v_a || !h_v_b || !h_v_output || !h_v_output_optimized || !h_lse_a || !h_lse_b || !h_lse_output || !h_lse_output_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_v_a, test_case.V_a, v_size);
        memcpy(h_v_b, test_case.V_b, v_size);
        memcpy(h_lse_a, test_case.LSE_a, lse_size);
        memcpy(h_lse_b, test_case.LSE_b, lse_size);

        // GPU memory allocation
        __nv_bfloat16 *d_v_a, *d_v_b, *d_v_output, *d_v_output_optimized;
        __nv_bfloat16 *d_lse_a, *d_lse_b, *d_lse_output, *d_lse_output_optimized;

        checkCudaError(cudaMalloc((void**)&d_v_a, v_size), "Allocating d_v_a");
        checkCudaError(cudaMalloc((void**)&d_v_b, v_size), "Allocating d_v_b");
        checkCudaError(cudaMalloc((void**)&d_v_output, v_size), "Allocating d_v_output");
        checkCudaError(cudaMalloc((void**)&d_v_output_optimized, v_size), "Allocating d_v_output_optimized");
        checkCudaError(cudaMalloc((void**)&d_lse_a, lse_size), "Allocating d_lse_a");
        checkCudaError(cudaMalloc((void**)&d_lse_b, lse_size), "Allocating d_lse_b");
        checkCudaError(cudaMalloc((void**)&d_lse_output, lse_size), "Allocating d_lse_output");
        checkCudaError(cudaMalloc((void**)&d_lse_output_optimized, lse_size), "Allocating d_lse_output_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_v_a, h_v_a, v_size, cudaMemcpyHostToDevice), "Copying h_v_a to d_v_a");
        checkCudaError(cudaMemcpy(d_v_b, h_v_b, v_size, cudaMemcpyHostToDevice), "Copying h_v_b to d_v_b");
        checkCudaError(cudaMemcpy(d_lse_a, h_lse_a, lse_size, cudaMemcpyHostToDevice), "Copying h_lse_a to d_lse_a");
        checkCudaError(cudaMemcpy(d_lse_b, h_lse_b, lse_size, cudaMemcpyHostToDevice), "Copying h_lse_b to d_lse_b");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */
        
        auto origin_kernel = [&]() {
            merge_attn_states_bf16_origin(d_v_output, d_lse_output, d_v_a, d_lse_a, d_v_b, d_lse_b, 
                            test_case.num_tokens, test_case.num_heads, test_case.head_size);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            merge_attn_states_bf16_optimized(d_v_output_optimized, d_lse_output_optimized, d_v_a, d_lse_a, d_v_b, d_lse_b,
                                    test_case.num_tokens, test_case.num_heads, test_case.head_size);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_v_output, d_v_output, v_size, cudaMemcpyDeviceToHost), "Copying d_v_output to h_v_output");
        checkCudaError(cudaMemcpy(h_v_output_optimized, d_v_output_optimized, v_size, cudaMemcpyDeviceToHost), "Copying d_v_output_optimized to h_v_output_optimized");
        checkCudaError(cudaMemcpy(h_lse_output, d_lse_output, lse_size, cudaMemcpyDeviceToHost), "Copying d_lse_output to h_lse_output");
        checkCudaError(cudaMemcpy(h_lse_output_optimized, d_lse_output_optimized, lse_size, cudaMemcpyDeviceToHost), "Copying d_lse_output_optimized to h_lse_output_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < v_item; i++) {
            if (!bfloat16_equals(h_v_output[i], h_v_output_optimized[i], 1e-2f)) {
                printf("V Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_v_output[i]), __bfloat162float(h_v_output_optimized[i]));
                return 1;
            }
        }

        for (int i = 0; i < lse_item; i++) {
            if (!bfloat16_equals(h_lse_output[i], h_lse_output_optimized[i], 1e-2f)) {
                printf("LSE Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_lse_output[i]), __bfloat162float(h_lse_output_optimized[i]));
                return 1;
            }
        }
        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_v_a);
        cudaFree(d_v_b);
        cudaFree(d_v_output);
        cudaFree(d_v_output_optimized);
        cudaFree(d_lse_a);
        cudaFree(d_lse_b);
        cudaFree(d_lse_output);
        cudaFree(d_lse_output_optimized);

        free(h_v_a);
        free(h_v_b);
        free(h_v_output);
        free(h_v_output_optimized);
        free(h_lse_a);
        free(h_lse_b);
        free(h_lse_output);
        free(h_lse_output_optimized);

        delete [] test_case.V_a;
        delete [] test_case.V_b;
        delete [] test_case.LSE_a;
        delete [] test_case.LSE_b;
    }

    return 0;
}