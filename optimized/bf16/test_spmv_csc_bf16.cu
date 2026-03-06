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
// Warp-centric CSR SpMV kernel: one warp per row, coalesced loads, read-only cache
__global__ void csr_spmv_kernel_optimized(
    int rows,
    const int* __restrict__ row_ptr,
    const int* __restrict__ cols,
    const __nv_bfloat16* __restrict__ values,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ y)
{
    const int warp_size = 32;
    int warps_per_block = blockDim.x / warp_size;
    int warp_id_in_block = threadIdx.x / warp_size;
    int warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    if (warp_id >= rows) return;

    int row = warp_id;
    int lane = threadIdx.x & (warp_size - 1);
    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    float sum = 0.0f;
    // Coalesced loads using read-only cache
    for (int idx = row_start + lane; idx < row_end; idx += warp_size) {
        __nv_bfloat16 v_bf = __ldg(&values[idx]);
        int col = __ldg(&cols[idx]);
        __nv_bfloat16 x_bf = __ldg(&x[col]);
        sum += __bfloat162float(v_bf) * __bfloat162float(x_bf);
    }

    // Warp-wide reduction using shuffle
    unsigned mask = 0xffffffff;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Write result by lane 0 of each warp
    if (lane == 0) {
        y[row] = sum;
    }
}

// Convert float array to bfloat16
__global__ void convert_float_to_bf16_kernel_optimized(
    const float* __restrict__ float_array,
    __nv_bfloat16* __restrict__ bf16_array,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmv_csc_bf16_optimized(
    int rows,
    int columns,
    const __nv_bfloat16* d_values,
    const int* d_row_indices,
    const int* d_col_offsets,
    const __nv_bfloat16* d_x,
    __nv_bfloat16* d_y)
{
    // Copy CSC structure from device to host
    int h_nnz;
    int* h_col_offsets = (int*)malloc((columns + 1) * sizeof(int));
    cudaMemcpy(h_col_offsets, d_col_offsets, (columns + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    h_nnz = h_col_offsets[columns];

    int* h_row_indices = (int*)malloc(h_nnz * sizeof(int));
    __nv_bfloat16* h_values = (__nv_bfloat16*)malloc(h_nnz * sizeof(__nv_bfloat16));
    cudaMemcpy(h_row_indices, d_row_indices, h_nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, d_values, h_nnz * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Build CSR on host
    int* h_row_counts = (int*)calloc(rows, sizeof(int));
    for (int i = 0; i < h_nnz; ++i) {
        int r = h_row_indices[i];
        h_row_counts[r]++;
    }
    int* h_csr_row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    h_csr_row_ptr[0] = 0;
    for (int i = 0; i < rows; ++i) {
        h_csr_row_ptr[i + 1] = h_csr_row_ptr[i] + h_row_counts[i];
    }
    int* h_csr_cols    = (int*)malloc(h_nnz * sizeof(int));
    __nv_bfloat16* h_csr_values = (__nv_bfloat16*)malloc(h_nnz * sizeof(__nv_bfloat16));
    int* h_position    = (int*)malloc(rows * sizeof(int));
    memcpy(h_position, h_csr_row_ptr, rows * sizeof(int));

    for (int col = 0; col < columns; ++col) {
        int start = h_col_offsets[col];
        int end   = h_col_offsets[col + 1];
        for (int j = start; j < end; ++j) {
            int row = h_row_indices[j];
            int dest = h_position[row]++;
            h_csr_cols[dest]    = col;
            h_csr_values[dest]  = h_values[j];
        }
    }

    // Allocate and copy CSR to device
    int* d_csr_row_ptr = nullptr;
    int* d_csr_cols    = nullptr;
    __nv_bfloat16* d_csr_values = nullptr;
    cudaMalloc(&d_csr_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_csr_cols,    h_nnz * sizeof(int));
    cudaMalloc(&d_csr_values,  h_nnz * sizeof(__nv_bfloat16));
    cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_cols,    h_csr_cols,    h_nnz * sizeof(int),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_values,  h_csr_values,  h_nnz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Free host CSR temp arrays
    free(h_col_offsets);
    free(h_row_indices);
    free(h_values);
    free(h_row_counts);
    free(h_csr_row_ptr);
    free(h_csr_cols);
    free(h_csr_values);
    free(h_position);

    // Allocate output float vector
    float* d_y_float = nullptr;
    cudaMalloc(&d_y_float, rows * sizeof(float));
    cudaMemset(d_y_float, 0, rows * sizeof(float));

    // Launch warp-centric CSR SpMV kernel
    const int warps_per_block = 4;
    int block_size = warps_per_block * 32;
    int grid_size  = (rows + warps_per_block - 1) / warps_per_block;
    csr_spmv_kernel_optimized<<<grid_size, block_size>>>(
        rows,
        d_csr_row_ptr,
        d_csr_cols,
        d_csr_values,
        d_x,
        d_y_float
    );

    // Convert float result to bfloat16
    int conv_grid = (rows + block_size - 1) / block_size;
    convert_float_to_bf16_kernel_optimized<<<conv_grid, block_size>>>(
        d_y_float,
        d_y,
        rows
    );

    // Free device memory
    cudaFree(d_csr_row_ptr);
    cudaFree(d_csr_cols);
    cudaFree(d_csr_values);
    cudaFree(d_y_float);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void spmv_csc_float_kernel(
    int columns,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_offsets,
    const __nv_bfloat16* x,
    float* y_float)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < columns) {
        float x_val_f = __bfloat162float(x[col]);
        int col_start = col_offsets[col];
        int col_end = col_offsets[col + 1];
        
        for (int j = col_start; j < col_end; j++) {
            int row = row_indices[j];
            float val_f = __bfloat162float(values[j]);

            float result_f = val_f * x_val_f;
            atomicAdd(&y_float[row], result_f);
        }
    }
}

__global__ void convert_float_to_bf16_kernel(float* float_array, __nv_bfloat16* bf16_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmv_csc_bf16_origin(
    int rows,
    int columns,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_offsets,
    const __nv_bfloat16* x,
    __nv_bfloat16* y) {
    
    float* d_y_float;
    cudaMalloc(&d_y_float, rows * sizeof(float));
    cudaMemset(d_y_float, 0, rows * sizeof(float));
    
    dim3 block(256, 1, 1);
    dim3 grid((columns + block.x - 1) / block.x, 1, 1);
    
    spmv_csc_float_kernel<<<grid, block>>>(
        columns, values, row_indices, col_offsets, x, d_y_float);
    
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
    int* row_indices;
    int* col_offsets;
    __nv_bfloat16* x;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> col_list = {1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15};
    int rows = 2048;
    float density = 0.01f;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    
    for (int i = 0; i < col_list.size(); i++) {        
        TestCase test_case;
        test_case.rows = rows;
        test_case.cols = col_list[i];
        
        int total_elements = test_case.rows * test_case.cols;
        test_case.nnz = static_cast<int>(total_elements * density);

        if (test_case.nnz < test_case.cols) {
            test_case.nnz = test_case.cols;
        }
        
        test_case.values = new __nv_bfloat16[test_case.nnz];
        test_case.row_indices = new int[test_case.nnz];
        test_case.col_offsets = new int[test_case.cols + 1];
        test_case.x = new __nv_bfloat16[test_case.cols];

        int nnz_per_col = test_case.nnz / test_case.cols;
        int remainder = test_case.nnz % test_case.cols;
        
        int current_offset = 0;
        for (int col = 0; col < test_case.cols; col++) {
            int col_nnz = nnz_per_col + (col < remainder ? 1 : 0);
    
            test_case.col_offsets[col] = current_offset;
            
            for (int k = 0; k < col_nnz; k++) {
                int idx = current_offset + k;
                test_case.values[idx] = __float2bfloat16(value_dist(rng));
                test_case.row_indices[idx] = row_dist(rng);
            }
            
            current_offset += col_nnz;
        }
        test_case.col_offsets[test_case.cols] = current_offset;
        
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
        const int col_offset_item = test_case.cols + 1;
        const int value_item = test_case.nnz;
        const int row_index_item = test_case.nnz;
        const int x_item = test_case.cols;
        const int y_item = test_case.rows;
        size_t col_offset_size = col_offset_item * sizeof(int);
        size_t value_size = value_item * sizeof(__nv_bfloat16);
        size_t row_index_size = row_index_item * sizeof(int);
        size_t x_size = x_item * sizeof(__nv_bfloat16);
        size_t y_size = y_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        int* h_col_offset = (int*)malloc(col_offset_size);
        __nv_bfloat16* h_values = (__nv_bfloat16*)malloc(value_size);
        int* h_row_index = (int*)malloc(row_index_size);
        __nv_bfloat16* h_x = (__nv_bfloat16*)malloc(x_size);
        __nv_bfloat16* h_y = (__nv_bfloat16*)malloc(y_size);
        __nv_bfloat16* h_y_optimized = (__nv_bfloat16*)malloc(y_size);

        if (!h_col_offset || !h_values || !h_row_index || !h_x || !h_y || !h_y_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_col_offset, test_case.col_offsets, col_offset_size);
        memcpy(h_values, test_case.values, value_size);
        memcpy(h_row_index, test_case.row_indices, row_index_size);
        memcpy(h_x, test_case.x, x_size);

        // GPU memory allocation
        int *d_col_offset, *d_row_index;
        __nv_bfloat16 *d_values, *d_x, *d_y, *d_y_optimized;

        checkCudaError(cudaMalloc((void**)&d_col_offset, col_offset_size), "Allocating d_col_offset");
        checkCudaError(cudaMalloc((void**)&d_row_index, row_index_size), "Allocating d_row_index");
        checkCudaError(cudaMalloc((void**)&d_values, value_size), "Allocating d_values");
        checkCudaError(cudaMalloc((void**)&d_x, x_size), "Allocating d_x");
        checkCudaError(cudaMalloc((void**)&d_y, y_size), "Allocating d_y");
        checkCudaError(cudaMalloc((void**)&d_y_optimized, y_size), "Allocating d_y_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_col_offset, h_col_offset, col_offset_size, cudaMemcpyHostToDevice), "Copying h_col_offset to d_col_offset");
        checkCudaError(cudaMemcpy(d_values, h_values, value_size, cudaMemcpyHostToDevice), "Copying h_values to d_values");
        checkCudaError(cudaMemcpy(d_row_index, h_row_index, row_index_size, cudaMemcpyHostToDevice), "Copying h_row_index to d_row_index");
        checkCudaError(cudaMemcpy(d_x, h_x, x_size, cudaMemcpyHostToDevice), "Copying h_x to d_x");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            cudaMemset(d_y, 0, y_size);
            spmv_csc_bf16_origin(test_case.rows, test_case.cols, d_values, d_row_index, d_col_offset, d_x, d_y);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            cudaMemset(d_y_optimized, 0, y_size);
            spmv_csc_bf16_optimized(test_case.rows, test_case.cols, d_values, d_row_index, d_col_offset, d_x, d_y_optimized);
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
        cudaFree(d_col_offset);
        cudaFree(d_row_index);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_y_optimized);

        free(h_col_offset);
        free(h_row_index);
        free(h_values);
        free(h_x);
        free(h_y);
        free(h_y_optimized);
        delete [] test_case.col_offsets;
        delete [] test_case.values;
        delete [] test_case.row_indices;
        delete [] test_case.x;
    }

    return 0;
}