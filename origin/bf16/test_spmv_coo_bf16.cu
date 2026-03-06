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
__global__ void spmv_coo_float_kernel_optimized(
    int nnz,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* x,
    float* y_float)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        float val_f = __bfloat162float(values[idx]);
        int row = row_indices[idx];
        int col = col_indices[idx];

        float x_val_f = __bfloat162float(x[col]);
        float result_f = val_f * x_val_f;
        
        atomicAdd(&y_float[row], result_f);
    }
}

__global__ void convert_float_to_bf16_kernel_optimized(float* float_array, __nv_bfloat16* bf16_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmv_coo_bf16_optimized(
    int rows,
    int nnz,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* x,
    __nv_bfloat16* y) {
    
    float* d_y_float;
    cudaMalloc(&d_y_float, rows * sizeof(float));
    cudaMemset(d_y_float, 0, rows * sizeof(float));
    
    dim3 block(256, 1, 1);
    dim3 grid((nnz + block.x - 1) / block.x, 1, 1);
    
    spmv_coo_float_kernel_optimized<<<grid, block>>>(
        nnz, values, row_indices, col_indices, x, d_y_float);
    
    dim3 convert_block(256);
    dim3 convert_grid((rows + convert_block.x - 1) / convert_block.x);
    convert_float_to_bf16_kernel_optimized<<<convert_grid, convert_block>>>(d_y_float, y, rows);
    
    cudaFree(d_y_float);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void spmv_coo_float_kernel(
    int nnz,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* x,
    float* y_float)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        float val_f = __bfloat162float(values[idx]);
        int row = row_indices[idx];
        int col = col_indices[idx];

        float x_val_f = __bfloat162float(x[col]);
        float result_f = val_f * x_val_f;
        
        atomicAdd(&y_float[row], result_f);
    }
}

__global__ void convert_float_to_bf16_kernel(float* float_array, __nv_bfloat16* bf16_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmv_coo_bf16_origin(
    int rows,
    int nnz,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* x,
    __nv_bfloat16* y) {
    
    float* d_y_float;
    cudaMalloc(&d_y_float, rows * sizeof(float));
    cudaMemset(d_y_float, 0, rows * sizeof(float));
    
    dim3 block(256, 1, 1);
    dim3 grid((nnz + block.x - 1) / block.x, 1, 1);
    
    spmv_coo_float_kernel<<<grid, block>>>(
        nnz, values, row_indices, col_indices, x, d_y_float);
    
    dim3 convert_block(256);
    dim3 convert_grid((rows + convert_block.x - 1) / convert_block.x);
    convert_float_to_bf16_kernel<<<convert_grid, convert_block>>>(d_y_float, y, rows);
    
    cudaFree(d_y_float);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int rows;
    int cols;
    int nnz;
    __nv_bfloat16* values;
    int* col_indices;
    int* row_indices;
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
        test_case.row_indices = new int[test_case.nnz];
        test_case.x = new __nv_bfloat16[test_case.cols];

        std::uniform_int_distribution<int> row_dist(0, test_case.rows - 1);

        for (int j = 0; j < test_case.rows; j++) {
            test_case.values[j] = __float2bfloat16(value_dist(rng));
            test_case.col_indices[j] = col_dist(rng);
            test_case.row_indices[j] = j;
        }
        
        for (int j = test_case.rows; j < test_case.nnz; j++) {
            test_case.values[j] = __float2bfloat16(value_dist(rng));
            test_case.col_indices[j] = col_dist(rng);
            test_case.row_indices[j] = row_dist(rng);
        }
        
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
        const int value_item = test_case.nnz;
        const int row_index_item = test_case.nnz;
        const int col_index_item = test_case.nnz;
        const int x_item = test_case.cols;
        const int y_item = test_case.rows;
        size_t row_index_size = row_index_item * sizeof(int);
        size_t col_index_size = col_index_item * sizeof(int);
        size_t value_size = value_item * sizeof(__nv_bfloat16);
        size_t x_size = x_item * sizeof(__nv_bfloat16);
        size_t y_size = y_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_values = (__nv_bfloat16*)malloc(value_size);
        int* h_row_index = (int*)malloc(row_index_size);
        int* h_col_index = (int*)malloc(col_index_size);
        __nv_bfloat16* h_x = (__nv_bfloat16*)malloc(x_size);
        __nv_bfloat16* h_y = (__nv_bfloat16*)malloc(y_size);
        __nv_bfloat16* h_y_optimized = (__nv_bfloat16*)malloc(y_size);

        if (!h_values || !h_row_index || !h_col_index || !h_x || !h_y || !h_y_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_values, test_case.values, value_size);
        memcpy(h_row_index, test_case.row_indices, row_index_size);
        memcpy(h_col_index, test_case.col_indices, col_index_size);
        memcpy(h_x, test_case.x, x_size);

        // GPU memory allocation
        int *d_row_index, *d_col_index;
        __nv_bfloat16 *d_values, *d_x, *d_y, *d_y_optimized;

        checkCudaError(cudaMalloc((void**)&d_row_index, row_index_size), "Allocating d_row_index");
        checkCudaError(cudaMalloc((void**)&d_col_index, col_index_size), "Allocating d_col_index");
        checkCudaError(cudaMalloc((void**)&d_values, value_size), "Allocating d_values");
        checkCudaError(cudaMalloc((void**)&d_x, x_size), "Allocating d_x");
        checkCudaError(cudaMalloc((void**)&d_y, y_size), "Allocating d_y");
        checkCudaError(cudaMalloc((void**)&d_y_optimized, y_size), "Allocating d_y_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_values, h_values, value_size, cudaMemcpyHostToDevice), "Copying h_values to d_values");
        checkCudaError(cudaMemcpy(d_row_index, h_row_index, row_index_size, cudaMemcpyHostToDevice), "Copying h_row_index to d_row_index");
        checkCudaError(cudaMemcpy(d_col_index, h_col_index, col_index_size, cudaMemcpyHostToDevice), "Copying h_col_index to d_col_index");
        checkCudaError(cudaMemcpy(d_x, h_x, x_size, cudaMemcpyHostToDevice), "Copying h_x to d_x");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            cudaMemset(d_y, 0, y_size);
            spmv_coo_bf16_origin(test_case.rows, test_case.nnz, d_values, d_row_index, d_col_index, d_x, d_y);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            cudaMemset(d_y_optimized, 0, y_size);
            spmv_coo_bf16_optimized(test_case.rows, test_case.nnz, d_values, d_row_index, d_col_index, d_x, d_y_optimized);
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
        cudaFree(d_row_index);
        cudaFree(d_col_index);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_y_optimized);

        free(h_row_index);
        free(h_col_index);
        free(h_values);
        free(h_x);
        free(h_y);
        free(h_y_optimized);
        delete [] test_case.values;
        delete [] test_case.row_indices;
        delete [] test_case.col_indices;
        delete [] test_case.x;
    }

    return 0;
}