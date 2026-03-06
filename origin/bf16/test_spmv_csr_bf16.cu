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
__global__ void spmv_csr_bf16_kernel_optimized(
    int rows,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* x,
    __nv_bfloat16* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];
        
        for (int j = row_start; j < row_end; j++) {
            float val_f = __bfloat162float(values[j]);
            float x_val_f = __bfloat162float(x[col_indices[j]]);
            sum += val_f * x_val_f;
        }
        
        y[row] = __float2bfloat16(sum);
    }
}

extern "C" void spmv_csr_bf16_optimized(
    int rows,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* x,
    __nv_bfloat16* y) {
    dim3 block(32, 32, 1);
    dim3 grid((rows + 31) / 32, 1, 1);
    
    spmv_csr_bf16_kernel_optimized<<<grid, block>>>(rows, values, col_indices, row_offsets, x, y);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void spmv_csr_bf16_kernel(
    int rows,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* x,
    __nv_bfloat16* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];
        
        for (int j = row_start; j < row_end; j++) {
            float val_f = __bfloat162float(values[j]);
            float x_val_f = __bfloat162float(x[col_indices[j]]);
            sum += val_f * x_val_f;
        }
        
        y[row] = __float2bfloat16(sum);
    }
}

extern "C" void spmv_csr_bf16_origin(
    int rows,
    const __nv_bfloat16* values,
    const int* col_indices,
    const int* row_offsets,
    const __nv_bfloat16* x,
    __nv_bfloat16* y) {
    dim3 block(32, 32, 1);
    dim3 grid((rows + 31) / 32, 1, 1);
    
    spmv_csr_bf16_kernel<<<grid, block>>>(rows, values, col_indices, row_offsets, x, y);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int rows;
    int cols;
    int nnz;
    __nv_bfloat16* values;
    int* col_indices;
    int* row_offsets;
    __nv_bfloat16* x;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> row_list = {1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15};
    int cols = 2048;
    float density = 0.01f;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    
    for (int i = 0; i < row_list.size(); i++) {        
        TestCase test_case;
        test_case.rows = row_list[i];
        test_case.cols = cols;
        
        int total_elements = test_case.rows * test_case.cols;
        test_case.nnz = static_cast<int>(total_elements * density);

        if (test_case.nnz < test_case.rows) {
            test_case.nnz = test_case.rows;
        }
        
        test_case.values = new __nv_bfloat16[test_case.nnz];
        test_case.col_indices = new int[test_case.nnz];
        test_case.row_offsets = new int[test_case.rows + 1];
        test_case.x = new __nv_bfloat16[test_case.cols];

        int nnz_per_row = test_case.nnz / test_case.rows;
        int remainder = test_case.nnz % test_case.rows;
        
        int current_offset = 0;
        for (int row = 0; row < test_case.rows; row++) {
            int row_nnz = nnz_per_row + (row < remainder ? 1 : 0);
    
            test_case.row_offsets[row] = current_offset;
            
            for (int k = 0; k < row_nnz; k++) {
                int idx = current_offset + k;
                test_case.values[idx] = __float2bfloat16(value_dist(rng));
                test_case.col_indices[idx] = col_dist(rng);
            }
            
            current_offset += row_nnz;
        }
        test_case.row_offsets[test_case.rows] = current_offset;
        
        for (int j = 0; j < test_case.cols; j++) {
            test_case.x[j] = __float2bfloat16(value_dist(rng));
        }
        
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: rows: %d, cols: %d, nnz: %d. Complexity: %d\n", test_case.rows, test_case.cols, test_case.nnz, test_case.nnz);
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
        const int row_offset_item = test_case.rows + 1;
        const int value_item = test_case.nnz;
        const int col_index_item = test_case.nnz;
        const int x_item = test_case.cols;
        const int y_item = test_case.rows;
        size_t row_offset_size = row_offset_item * sizeof(int);
        size_t value_size = value_item * sizeof(__nv_bfloat16);
        size_t col_index_size = col_index_item * sizeof(int);
        size_t x_size = x_item * sizeof(__nv_bfloat16);
        size_t y_size = y_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        int* h_row_offset = (int*)malloc(row_offset_size);
        __nv_bfloat16* h_values = (__nv_bfloat16*)malloc(value_size);
        int* h_col_index = (int*)malloc(col_index_size);
        __nv_bfloat16* h_x = (__nv_bfloat16*)malloc(x_size);
        __nv_bfloat16* h_y = (__nv_bfloat16*)malloc(y_size);
        __nv_bfloat16* h_y_optimized = (__nv_bfloat16*)malloc(y_size);

        if (!h_row_offset || !h_values || !h_col_index || !h_x || !h_y || !h_y_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_row_offset, test_case.row_offsets, row_offset_size);
        memcpy(h_values, test_case.values, value_size);
        memcpy(h_col_index, test_case.col_indices, col_index_size);
        memcpy(h_x, test_case.x, x_size);

        // GPU memory allocation
        int *d_row_offset, *d_col_index;
        __nv_bfloat16 *d_values, *d_x, *d_y, *d_y_optimized;

        checkCudaError(cudaMalloc((void**)&d_row_offset, row_offset_size), "Allocating d_row_offset");
        checkCudaError(cudaMalloc((void**)&d_col_index, col_index_size), "Allocating d_col_index");
        checkCudaError(cudaMalloc((void**)&d_values, value_size), "Allocating d_values");
        checkCudaError(cudaMalloc((void**)&d_x, x_size), "Allocating d_x");
        checkCudaError(cudaMalloc((void**)&d_y, y_size), "Allocating d_y");
        checkCudaError(cudaMalloc((void**)&d_y_optimized, y_size), "Allocating d_y_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_row_offset, h_row_offset, row_offset_size, cudaMemcpyHostToDevice), "Copying h_row_offset to d_row_offset");
        checkCudaError(cudaMemcpy(d_values, h_values, value_size, cudaMemcpyHostToDevice), "Copying h_values to d_values");
        checkCudaError(cudaMemcpy(d_col_index, h_col_index, col_index_size, cudaMemcpyHostToDevice), "Copying h_col_index to d_col_index");
        checkCudaError(cudaMemcpy(d_x, h_x, x_size, cudaMemcpyHostToDevice), "Copying h_x to d_x");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            spmv_csr_bf16_origin(test_case.rows, d_values, d_col_index, d_row_offset, d_x, d_y);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            spmv_csr_bf16_optimized(test_case.rows, d_values, d_col_index, d_row_offset, d_x, d_y_optimized);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_y, d_y, y_size, cudaMemcpyDeviceToHost), "Copying d_y to h_y");
        checkCudaError(cudaMemcpy(h_y_optimized, d_y_optimized, y_size, cudaMemcpyDeviceToHost), "Copying d_y_optimized to h_y_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < y_item; i++) {
            if (!bfloat16_equals(h_y[i], h_y_optimized[i], 1e-1f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", 
                       i, __bfloat162float(h_y[i]), __bfloat162float(h_y_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_row_offset);
        cudaFree(d_col_index);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_y_optimized);

        free(h_row_offset);
        free(h_values);
        free(h_col_index);
        free(h_x);
        free(h_y);
        free(h_y_optimized);
        delete [] test_case.row_offsets;
        delete [] test_case.values;
        delete [] test_case.col_indices;
        delete [] test_case.x;
    }

    return 0;
}