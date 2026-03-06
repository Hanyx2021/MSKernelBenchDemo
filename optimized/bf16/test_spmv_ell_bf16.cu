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
// Optimized ELL-format SpMV kernel using warp-striping and bfloat16 values
__global__ void spmv_ell_bf16_kernel_optimized(
    int rows,
    int max_nnz_per_row,
    const __nv_bfloat16 *__restrict__ values,
    const int *__restrict__ col_ids,
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ y) {
    // Compute warp and lane identifiers
    int lane_id = threadIdx.x & 31;
    int warp_id_in_block = threadIdx.x >> 5;               // threadIdx.x / 32
    int warps_per_block = blockDim.x >> 5;                 // blockDim.x / 32
    int warp_global_id = blockIdx.x * warps_per_block + warp_id_in_block;

    // Each warp works on one row
    if (warp_global_id >= rows) return;
    int row = warp_global_id;

    float lane_sum = 0.0f;
    // Warp-striping over non-zero entries for full coalescing
    for (int idx = lane_id; idx < max_nnz_per_row; idx += 32) {
        int offset = row * max_nnz_per_row + idx;
        // Load value and column index via read-only cache
        __nv_bfloat16 bf16_v = __ldg(&values[offset]);
        int c = __ldg(&col_ids[offset]);
        if (c >= 0) {
            float v_f = __bfloat162float(bf16_v);
            float x_f = __bfloat162float(__ldg(&x[c]));
            lane_sum += v_f * x_f;
        }
    }

    // Warp-synchronous reduction of lane sums
    for (int s = 16; s > 0; s >>= 1) {
        lane_sum += __shfl_down_sync(0xFFFFFFFF, lane_sum, s);
    }

    // Write result by lane 0 of each warp
    if (lane_id == 0) {
        y[row] = __float2bfloat16(lane_sum);
    }
}

extern "C" void spmv_ell_bf16_optimized(
    int rows,
    int max_nnz_per_row,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *x,
    __nv_bfloat16 *y) {
    // Configure launch: 256 threads per block (8 warps)
    const int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    dim3 block(threads_per_block, 1, 1);
    dim3 grid((rows + warps_per_block - 1) / warps_per_block, 1, 1);

    spmv_ell_bf16_kernel_optimized<<<grid, block>>>(
        rows, max_nnz_per_row, values, col_ids, x, y);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void spmv_ell_bf16_kernel(
    int rows,
    int max_nnz_per_row,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *x,
    __nv_bfloat16 *y) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        
        for (int element = 0; element < max_nnz_per_row; element++) {
            int offset = row * max_nnz_per_row + element;
            
            int col = col_ids[offset];
            if (col != -1) {
                float val_f = __bfloat162float(values[offset]);
                float x_val_f = __bfloat162float(x[col]);
                sum += val_f * x_val_f;
            }
        }
        
        y[row] = __float2bfloat16(sum);
    }
}

extern "C" void spmv_ell_bf16_origin(
    int rows,
    int max_nnz_per_row,
    const __nv_bfloat16 *values,
    const int *col_ids,
    const __nv_bfloat16 *x,
    __nv_bfloat16 *y) {
    dim3 block(256, 1, 1);
    dim3 grid((rows + block.x - 1) / block.x, 1, 1);
    
    spmv_ell_bf16_kernel<<<grid, block>>>(rows, max_nnz_per_row, values, col_ids, x, y);
    cudaDeviceSynchronize();
}


// Test case input data structure
typedef struct {
    int rows;
    int cols;
    int nnz;
    int max_nnz_per_row;
    __nv_bfloat16* values;
    int* col_ids;
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
    
    for (int i = 0; i < col_list.size(); i++) {        
        TestCase test_case;
        test_case.rows = rows;
        test_case.cols = col_list[i];
        
        int total_elements = test_case.rows * test_case.cols;
        test_case.nnz = static_cast<int>(total_elements * density);
        
        if (test_case.nnz < test_case.rows) {
            test_case.nnz = test_case.rows;
        }
        
        std::vector<std::vector<int>> row_cols(test_case.rows);
        std::vector<std::vector<__nv_bfloat16>> row_vals(test_case.rows);
        
        std::uniform_int_distribution<int> row_dist(0, test_case.rows - 1);
        std::uniform_int_distribution<int> col_dist(0, test_case.cols - 1);
        
        for (int row = 0; row < test_case.rows; row++) {
            int col = col_dist(rng);
            row_cols[row].push_back(col);
            row_vals[row].push_back(__float2bfloat16(value_dist(rng)));
        }
        
        int remaining_nnz = test_case.nnz - test_case.rows;
        for (int j = 0; j < remaining_nnz; j++) {
            int row = row_dist(rng);
            int col = col_dist(rng);
            row_cols[row].push_back(col);
            row_vals[row].push_back(__float2bfloat16(value_dist(rng)));
        }
        
        int max_nnz_per_row = 0;
        for (int row = 0; row < test_case.rows; row++) {
            if (row_cols[row].size() > max_nnz_per_row) {
                max_nnz_per_row = row_cols[row].size();
            }
        }
        
        test_case.max_nnz_per_row = max_nnz_per_row;
        
        int ell_size = test_case.rows * test_case.max_nnz_per_row;
        test_case.values = new __nv_bfloat16[ell_size];
        test_case.col_ids = new int[ell_size];
        test_case.x = new __nv_bfloat16[test_case.cols];
        
        for (int j = 0; j < ell_size; j++) {
            test_case.values[j] = __float2bfloat16(0.0f);
            test_case.col_ids[j] = -1;
        }
        
        for (int row = 0; row < test_case.rows; row++) {
            int row_start = row * test_case.max_nnz_per_row;
            int num_elements = row_cols[row].size();
            
            for (int elem = 0; elem < num_elements; elem++) {
                int offset = row_start + elem;
                test_case.values[offset] = row_vals[row][elem];
                test_case.col_ids[offset] = row_cols[row][elem];
            }
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
        const int value_item = test_case.rows * test_case.max_nnz_per_row;
        const int col_ids_item = test_case.rows * test_case.max_nnz_per_row;
        const int x_item = test_case.cols;
        const int y_item = test_case.rows;
        size_t value_size = value_item * sizeof(__nv_bfloat16);
        size_t col_ids_size = col_ids_item * sizeof(int);
        size_t x_size = x_item * sizeof(__nv_bfloat16);
        size_t y_size = y_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_values = (__nv_bfloat16*)malloc(value_size);
        int* h_col_ids = (int*)malloc(col_ids_size);
        __nv_bfloat16* h_x = (__nv_bfloat16*)malloc(x_size);
        __nv_bfloat16* h_y = (__nv_bfloat16*)malloc(y_size);
        __nv_bfloat16* h_y_optimized = (__nv_bfloat16*)malloc(y_size);

        if (!h_values || !h_col_ids || !h_x || !h_y || !h_y_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_values, test_case.values, value_size);
        memcpy(h_col_ids, test_case.col_ids, col_ids_size);
        memcpy(h_x, test_case.x, x_size);

        // GPU memory allocation
        int *d_col_ids;
        __nv_bfloat16 *d_values, *d_x, *d_y, *d_y_optimized;

        checkCudaError(cudaMalloc((void**)&d_col_ids, col_ids_size), "Allocating d_col_ids");
        checkCudaError(cudaMalloc((void**)&d_values, value_size), "Allocating d_values");
        checkCudaError(cudaMalloc((void**)&d_x, x_size), "Allocating d_x");
        checkCudaError(cudaMalloc((void**)&d_y, y_size), "Allocating d_y");
        checkCudaError(cudaMalloc((void**)&d_y_optimized, y_size), "Allocating d_y_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_values, h_values, value_size, cudaMemcpyHostToDevice), "Copying h_values to d_values");
        checkCudaError(cudaMemcpy(d_col_ids, h_col_ids, col_ids_size, cudaMemcpyHostToDevice), "Copying h_col_ids to d_col_ids");
        checkCudaError(cudaMemcpy(d_x, h_x, x_size, cudaMemcpyHostToDevice), "Copying h_x to d_x");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            cudaMemset(d_y, 0, y_size);
            spmv_ell_bf16_origin(test_case.rows, test_case.max_nnz_per_row, d_values, d_col_ids, d_x, d_y);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            cudaMemset(d_y_optimized, 0, y_size);
            spmv_ell_bf16_optimized(test_case.rows, test_case.max_nnz_per_row, d_values, d_col_ids, d_x, d_y_optimized);
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
        cudaFree(d_col_ids);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_y_optimized);

        free(h_col_ids);
        free(h_values);
        free(h_x);
        free(h_y);
        free(h_y_optimized);
        delete [] test_case.values;
        delete [] test_case.col_ids;
        delete [] test_case.x;
    }

    return 0;
}