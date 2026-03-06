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
// Optimized CSR-BF16 SpMM Kernel with full-warp coalesced loads
__global__ void spmm_csr_bf16_kernel_optimized(
    int rows,
    int K,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y)
{
    // Each block covers blockDim.y rows and blockDim.x columns (32-wide for coalescing)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_k = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col_k < K) {
        float sum = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];

        for (int j = row_start; j < row_end; j++) {
            int col_index = col_indices[j];
            float val_f = __bfloat162float(values[j]);
            float x_val_f = __bfloat162float(X[col_index * K + col_k]);
            sum += val_f * x_val_f;
        }

        Y[row * K + col_k] = __float2bfloat16(sum);
    }
}

extern "C" void spmm_csr_bf16_optimized(
    int rows,
    int K,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y)
{
    // Launch configuration tuned for RTX 4090: 32 threads in x for full-warps, 8 threads in y
    dim3 block(32, 8);
    dim3 grid((K + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    spmm_csr_bf16_kernel_optimized<<<grid, block>>>(
        rows, K, values, col_indices, row_offsets, X, Y);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void spmm_csr_bf16_kernel(
    int rows,
    int K,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_k = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col_k < K) {
        float sum = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];

        for (int j = row_start; j < row_end; j++) {
            int col_index = col_indices[j];
            float val_f = __bfloat162float(values[j]);
            float x_val_f = __bfloat162float(X[col_index * K + col_k]);
            sum += val_f * x_val_f;
        }

        Y[row * K + col_k] = __float2bfloat16(sum);
    }
}

extern "C" void spmm_csr_bf16_origin(
    int rows,
    int K,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y) {
    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (rows + 15) / 16);
    
    spmm_csr_bf16_kernel<<<grid, block>>>(rows, K, values, col_indices, row_offsets, X, Y);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int rows;
    int cols;
    int K;
    int nnz;
    __nv_bfloat16* values;
    int* col_indices;
    int* row_offsets;
    __nv_bfloat16* X;        //   X (cols x K)
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> row_list = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};
    int cols = 2048; 
    int K = 4096;
    float density = 0.01f;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    
    for (int rows : row_list) {
        TestCase test_case;
        test_case.rows = rows;
        test_case.cols = cols;
        test_case.K = K;
        
        int total_elements = rows * cols;
        test_case.nnz = static_cast<int>(total_elements * density);
        
        if (test_case.nnz < rows) {
            test_case.nnz = rows;
        }
        
        test_case.values = new __nv_bfloat16[test_case.nnz];
        test_case.col_indices = new int[test_case.nnz];
        test_case.row_offsets = new int[rows + 1];
        test_case.X = new __nv_bfloat16[cols * K];
        
        int nnz_per_row = test_case.nnz / rows;
        int remainder = test_case.nnz % rows;
        
        int current_offset = 0;
        for (int row = 0; row < rows; row++) {
            int row_nnz = nnz_per_row + (row < remainder ? 1 : 0);
            test_case.row_offsets[row] = current_offset;
            
            for (int k = 0; k < row_nnz; k++) {
                int idx = current_offset + k;
                test_case.values[idx] = __float2bfloat16(value_dist(rng));
                test_case.col_indices[idx] = col_dist(rng);
            }
            
            current_offset += row_nnz;
        }
        test_case.row_offsets[rows] = current_offset;

        for (int j = 0; j < cols * K; j++) {
            test_case.X[j] = __float2bfloat16(value_dist(rng));
        }
        
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: rows: %d, cols: %d, K: %d, nnz: %d. Complexity: %d\n", test_case.rows, test_case.cols, test_case.K, test_case.nnz, test_case.nnz * test_case.K);
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
        const int row_offset_items = test_case.rows + 1;
        const int value_items = test_case.nnz;
        const int col_index_items = test_case.nnz;
        const int X_items = test_case.cols * test_case.K;
        const int Y_items = test_case.rows * test_case.K;
        
        size_t row_offset_size = row_offset_items * sizeof(int);
        size_t value_size = value_items * sizeof(__nv_bfloat16);
        size_t col_index_size = col_index_items * sizeof(int);
        size_t X_size = X_items * sizeof(__nv_bfloat16);
        size_t Y_size = Y_items * sizeof(__nv_bfloat16);

        // Host memory inputs
        int* h_row_offset = (int*)malloc(row_offset_size);
        __nv_bfloat16* h_values = (__nv_bfloat16*)malloc(value_size);
        int* h_col_index = (int*)malloc(col_index_size);
        __nv_bfloat16* h_X = (__nv_bfloat16*)malloc(X_size);
        __nv_bfloat16* h_Y = (__nv_bfloat16*)malloc(Y_size);
        __nv_bfloat16* h_Y_optimized = (__nv_bfloat16*)malloc(Y_size);

        if (!h_row_offset || !h_values || !h_col_index || !h_X || !h_Y || !h_Y_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_row_offset, test_case.row_offsets, row_offset_size);
        memcpy(h_values, test_case.values, value_size);
        memcpy(h_col_index, test_case.col_indices, col_index_size);
        memcpy(h_X, test_case.X, X_size);

        // GPU memory allocation
        int *d_row_offset, *d_col_index;
        __nv_bfloat16 *d_values, *d_X, *d_Y, *d_Y_optimized;

        checkCudaError(cudaMalloc((void**)&d_row_offset, row_offset_size), "Allocating d_row_offset");
        checkCudaError(cudaMalloc((void**)&d_col_index, col_index_size), "Allocating d_col_index");
        checkCudaError(cudaMalloc((void**)&d_values, value_size), "Allocating d_values");
        checkCudaError(cudaMalloc((void**)&d_X, X_size), "Allocating d_X");
        checkCudaError(cudaMalloc((void**)&d_Y, Y_size), "Allocating d_Y");
        checkCudaError(cudaMalloc((void**)&d_Y_optimized, Y_size), "Allocating d_Y_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_row_offset, h_row_offset, row_offset_size, cudaMemcpyHostToDevice), "Copying h_row_offset to d_row_offset");
        checkCudaError(cudaMemcpy(d_values, h_values, value_size, cudaMemcpyHostToDevice), "Copying h_values to d_values");
        checkCudaError(cudaMemcpy(d_col_index, h_col_index, col_index_size, cudaMemcpyHostToDevice), "Copying h_col_index to d_col_index");
        checkCudaError(cudaMemcpy(d_X, h_X, X_size, cudaMemcpyHostToDevice), "Copying h_X to d_X");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto original_kernel = [&]() {
            spmm_csr_bf16_origin(test_case.rows, test_case.K, 
                    d_values, d_col_index, d_row_offset, 
                    d_X, d_Y);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            spmm_csr_bf16_optimized(test_case.rows, test_case.K, 
                    d_values, d_col_index, d_row_offset, 
                    d_X, d_Y_optimized);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(original_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_Y, d_Y, Y_size, cudaMemcpyDeviceToHost), "Copying d_Y to h_Y");
        checkCudaError(cudaMemcpy(h_Y_optimized, d_Y_optimized, Y_size, cudaMemcpyDeviceToHost), "Copying d_Y_optimized to h_Y_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < Y_items; i++) {
            if (!bfloat16_equals(h_Y[i], h_Y_optimized[i], 1e-1f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", 
                       i, __bfloat162float(h_Y[i]), __bfloat162float(h_Y_optimized[i]));
                return 1;
            }
        }
        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_row_offset);
        cudaFree(d_col_index);
        cudaFree(d_values);
        cudaFree(d_X);
        cudaFree(d_Y);
        cudaFree(d_Y_optimized);

        free(h_row_offset);
        free(h_values);
        free(h_col_index);
        free(h_X);
        free(h_Y);
        free(h_Y_optimized);
        
        delete [] test_case.row_offsets;
        delete [] test_case.values;
        delete [] test_case.col_indices;
        delete [] test_case.X;
    }

    return 0;
}