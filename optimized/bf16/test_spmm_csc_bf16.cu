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
// Configuration struct for optimized kernel launches
typedef struct {
    dim3 spmm_block;
    dim3 spmm_grid;
    dim3 convert_block;
    dim3 convert_grid;
} SpmmCscBf16OptimizedConfig;

// CSR-based SpMM kernel: each thread computes one (row, k) output without atomics
// Warp-aligned mapping: threadIdx.x -> K dimension, threadIdx.y -> row dimension
__global__ void spmm_csr_float_kernel_optimized(
    int rows,
    int K,
    const __nv_bfloat16* __restrict__ values_csr,
    const int* __restrict__ col_indices_csr,
    const int* __restrict__ row_offsets_csr,
    const __nv_bfloat16* __restrict__ X,
    float* __restrict__ Y_float)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && k < K) {
        float sum = 0.0f;
        int start = __ldg(&row_offsets_csr[r]);
        int end   = __ldg(&row_offsets_csr[r + 1]);
        for (int idx = start; idx < end; ++idx) {
            int c = __ldg(&col_indices_csr[idx]);
            float a_val = __bfloat162float(__ldg(&values_csr[idx]));
            float x_val = __bfloat162float(__ldg(&X[c * K + k]));
            sum += a_val * x_val;
        }
        Y_float[r * K + k] = sum;
    }
}

// Optimized conversion from float array to bfloat16 array
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

// External C wrapper for the optimized operator
extern "C" void spmm_csc_bf16_optimized(
    int rows,
    int columns,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y)
{
    // Determine number of non-zeros (nnz)
    int h_nnz = 0;
    cudaMemcpy(&h_nnz, col_offsets + columns, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy CSC data to host
    std::vector<int> h_col_offsets(columns + 1);
    cudaMemcpy(h_col_offsets.data(), col_offsets, sizeof(int) * (columns + 1), cudaMemcpyDeviceToHost);
    std::vector<int> h_row_indices(h_nnz);
    cudaMemcpy(h_row_indices.data(), row_indices, sizeof(int) * h_nnz, cudaMemcpyDeviceToHost);
    std::vector<__nv_bfloat16> h_values(h_nnz);
    cudaMemcpy(h_values.data(), values, sizeof(__nv_bfloat16) * h_nnz, cudaMemcpyDeviceToHost);

    // Build CSR structure on host
    std::vector<int> h_row_offsets(rows + 1, 0);
    for (int col = 0; col < columns; ++col) {
        int start = h_col_offsets[col];
        int end   = h_col_offsets[col + 1];
        for (int j = start; j < end; ++j) {
            int r = h_row_indices[j];
            h_row_offsets[r + 1]++;
        }
    }
    for (int r = 1; r <= rows; ++r) {
        h_row_offsets[r] += h_row_offsets[r - 1];
    }
    
    std::vector<int> h_row_ptr = h_row_offsets;
    std::vector<int> h_col_indices_csr(h_nnz);
    std::vector<__nv_bfloat16> h_values_csr(h_nnz);
    for (int col = 0; col < columns; ++col) {
        int start = h_col_offsets[col];
        int end   = h_col_offsets[col + 1];
        for (int j = start; j < end; ++j) {
            int r = h_row_indices[j];
            int pos = h_row_ptr[r]++;
            h_col_indices_csr[pos] = col;
            h_values_csr[pos]      = h_values[j];
        }
    }

    // Allocate and copy CSR arrays to device
    int* d_row_offsets_csr = nullptr;
    int* d_col_indices_csr = nullptr;
    __nv_bfloat16* d_values_csr = nullptr;
    cudaMalloc(&d_row_offsets_csr, sizeof(int) * (rows + 1));
    cudaMalloc(&d_col_indices_csr, sizeof(int) * h_nnz);
    cudaMalloc(&d_values_csr, sizeof(__nv_bfloat16) * h_nnz);
    cudaMemcpy(d_row_offsets_csr, h_row_offsets.data(), sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices_csr, h_col_indices_csr.data(), sizeof(int) * h_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_csr, h_values_csr.data(), sizeof(__nv_bfloat16) * h_nnz, cudaMemcpyHostToDevice);

    // Allocate and zero temporary float output
    float* d_Y_float = nullptr;
    size_t total_floats = static_cast<size_t>(rows) * static_cast<size_t>(K);
    cudaMalloc(&d_Y_float, total_floats * sizeof(float));
    cudaMemset(d_Y_float, 0, total_floats * sizeof(float));

    // Configure and launch optimized spmm CSR kernel with warp-aligned mapping
    SpmmCscBf16OptimizedConfig cfg;
    cfg.spmm_block    = dim3(32, 8);
    cfg.spmm_grid     = dim3((K    + cfg.spmm_block.x - 1) / cfg.spmm_block.x,
                              (rows + cfg.spmm_block.y - 1) / cfg.spmm_block.y);
    cfg.convert_block = dim3(256);
    cfg.convert_grid  = dim3((total_floats + cfg.convert_block.x - 1) / cfg.convert_block.x);

    spmm_csr_float_kernel_optimized<<<cfg.spmm_grid, cfg.spmm_block>>>(
        rows, K,
        d_values_csr, d_col_indices_csr, d_row_offsets_csr,
        X, d_Y_float);

    // Convert accumulated floats back to bfloat16
    convert_float_to_bf16_kernel_optimized<<<cfg.convert_grid, cfg.convert_block>>>(
        d_Y_float, Y, static_cast<int>(total_floats));

    // Clean up and synchronize
    cudaFree(d_Y_float);
    cudaFree(d_row_offsets_csr);
    cudaFree(d_col_indices_csr);
    cudaFree(d_values_csr);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void spmm_csc_float_kernel(
    int columns,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_offsets,
    const __nv_bfloat16* X,
    float* Y_float)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < columns && k < K) {
        int col_start = col_offsets[col];
        int col_end = col_offsets[col + 1];
        
        for (int j = col_start; j < col_end; j++) {

            float val_f = __bfloat162float(values[j]);
            float x_val_f = __bfloat162float(X[col * K + k]);
            float result_f = val_f * x_val_f;
        
            int row = row_indices[j];
            atomicAdd(&Y_float[row * K + k], result_f);
        }
    }
}

__global__ void convert_float_to_bf16_kernel(float* float_array, __nv_bfloat16* bf16_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmm_csc_bf16_origin(
    int rows,
    int columns,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_offsets,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y) {

    float* d_Y_float;
    cudaMalloc(&d_Y_float, rows * K * sizeof(float));
    cudaMemset(d_Y_float, 0, rows * K * sizeof(float));

    dim3 block(16, 16, 1);
    dim3 grid((columns + block.x - 1) / block.x, (K + block.y - 1) / block.y, 1);
    
    spmm_csc_float_kernel<<<grid, block>>>(columns, K, values, row_indices, col_offsets, X, d_Y_float);

    dim3 convert_block(256);
    dim3 convert_grid((rows * K + convert_block.x - 1) / convert_block.x);
    convert_float_to_bf16_kernel<<<convert_grid, convert_block>>>(d_Y_float, Y, rows * K);
    
    cudaFree(d_Y_float);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int rows;
    int cols;
    int K;
    int nnz;
    __nv_bfloat16* values;
    int* row_indices;
    int* col_offsets;
    __nv_bfloat16* x;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> col_list = {1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14};
    int rows = 2048;
    int K = 4096;
    float density = 0.01f;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    
    for (int i = 0; i < col_list.size(); i++) {        
        TestCase test_case;
        test_case.rows = rows;
        test_case.cols = col_list[i];
        test_case.K = K;
        
        int total_elements = test_case.rows * test_case.cols;
        test_case.nnz = static_cast<int>(total_elements * density);

        if (test_case.nnz < test_case.cols) {
            test_case.nnz = test_case.cols;
        }
        
        test_case.values = new __nv_bfloat16[test_case.nnz];
        test_case.row_indices = new int[test_case.nnz];
        test_case.col_offsets = new int[test_case.cols + 1];
        test_case.x = new __nv_bfloat16[test_case.cols * K];

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
        
        for (int j = 0; j < test_case.cols * K; j++) {
            test_case.x[j] = __float2bfloat16(value_dist(rng));
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
        const int col_offset_item = test_case.cols + 1;
        const int value_item = test_case.nnz;
        const int row_index_item = test_case.nnz;
        const int x_item = test_case.cols * test_case.K;
        const int y_item = test_case.rows * test_case.K;
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
            spmm_csc_bf16_origin(test_case.rows, test_case.cols, test_case.K, d_values, d_row_index, d_col_offset, d_x, d_y);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            cudaMemset(d_y_optimized, 0, y_size);
            spmm_csc_bf16_optimized(test_case.rows, test_case.cols, test_case.K, d_values, d_row_index, d_col_offset, d_x, d_y_optimized);
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