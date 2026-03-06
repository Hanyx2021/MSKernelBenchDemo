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
// Tile size for K-dimension blocking
#define TILE 256

// Kernel to convert bfloat16 array to float array
__global__ void convert_bf16_to_float_kernel_optimized(
    const __nv_bfloat16* bf16_array,
    float* float_array,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float_array[idx] = __bfloat162float(bf16_array[idx]);
    }
}

// Kernel to convert float array back to bfloat16
__global__ void convert_float_to_bf16_kernel_optimized(
    const float* float_array,
    __nv_bfloat16* bf16_array,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

// CSR-based SpMM kernel operating on float data, one block per row, optimized with shared memory
__global__ void spmm_csr_kernel_optimized(
    int rows,
    int K,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_csr,
    const float* __restrict__ val_f,
    const float* __restrict__ X_f,
    float* Y_f) {
    extern __shared__ char smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int start = row_ptr[row];
    int end   = row_ptr[row + 1];
    int nnz   = end - start;

    // Shared-memory buffers for values and column indices
    float* s_val = (float*)smem;
    int*   s_col = (int*)(smem + nnz * sizeof(float));

    // Load sparse row into shared memory
    int t = threadIdx.x;
    if (t < nnz) {
        s_val[t] = val_f[start + t];
        s_col[t] = col_csr[start + t];
    }
    __syncthreads();

    // Iterate over tiles along K dimension
    for (int tile_start = 0; tile_start < K; tile_start += TILE) {
        int k_index = tile_start + t;
        if (k_index < K) {
            float acc = 0.0f;
            // Accumulate contributions for this tile using shared memory
            for (int j = 0; j < nnz; ++j) {
                float v = s_val[j];
                int   c = s_col[j];
                // Use read-only cache for dense input
                float x = __ldg(&X_f[c * (size_t)K + k_index]);
                acc += v * x;
            }
            Y_f[row * (size_t)K + k_index] = acc;
        }
        // no per-tile __syncthreads() needed
    }
}

extern "C" void spmm_coo_bf16_optimized(
    int rows,
    int nnz,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y) {
    // Copy COO data to host
    std::vector<int> h_row(nnz), h_col(nnz);
    std::vector<__nv_bfloat16> h_val_bf16(nnz);
    cudaMemcpy(h_row.data(), row_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col.data(), col_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val_bf16.data(), values,   nnz * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Build CSR row_ptr with exclusive prefix sum
    std::vector<int> h_row_ptr(rows + 1, 0);
    for (int i = 0; i < nnz; ++i) {
        int r = h_row[i];
        h_row_ptr[r + 1]++;
    }
    for (int i = 1; i <= rows; ++i) {
        h_row_ptr[i] += h_row_ptr[i - 1];
    }

    // Compute maximum non-zeros per row for shared memory sizing
    int max_nnz = 0;
    for (int r = 0; r < rows; ++r) {
        int count = h_row_ptr[r + 1] - h_row_ptr[r];
        if (count > max_nnz) max_nnz = count;
    }
    size_t shared_mem_size = max_nnz * (sizeof(float) + sizeof(int));

    // Scatter into CSR storage and convert values to float
    std::vector<int>   h_col_csr(nnz);
    std::vector<float> h_val_f(nnz);
    std::vector<int>   cur_pos = h_row_ptr;
    for (int i = 0; i < nnz; ++i) {
        int r = h_row[i];
        int pos = cur_pos[r]++;
        h_col_csr[pos] = h_col[i];
        h_val_f[pos]    = __bfloat162float(h_val_bf16[i]);
    }

    // Determine number of columns
    int max_col = 0;
    for (int c : h_col) max_col = std::max(max_col, c);
    int cols = max_col + 1;

    // Allocate and upload CSR arrays to device
    int* d_row_ptr = nullptr;
    int* d_col_csr = nullptr;
    float* d_val_f = nullptr;
    cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_col_csr, nnz * sizeof(int));
    cudaMalloc(&d_val_f,   nnz * sizeof(float));
    cudaMemcpy(d_row_ptr,  h_row_ptr.data(),     (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_csr,  h_col_csr.data(),     nnz * sizeof(int),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_f,    h_val_f.data(),       nnz * sizeof(float),      cudaMemcpyHostToDevice);

    // Convert input dense matrix X to float
    float* d_X_f = nullptr;
    cudaMalloc(&d_X_f, (size_t)cols * K * sizeof(float));
    const int block_convert = 256;
    int total_X = cols * K;
    int grid_convert_X = (total_X + block_convert - 1) / block_convert;
    convert_bf16_to_float_kernel_optimized<<<grid_convert_X, block_convert>>>(
        X, d_X_f, total_X);

    // Allocate output in float
    float* d_Y_f = nullptr;
    cudaMalloc(&d_Y_f, (size_t)rows * K * sizeof(float));

    // Launch CSR-based SpMM kernel with shared memory
    dim3 block(TILE);
    dim3 grid(rows);
    spmm_csr_kernel_optimized<<<grid, block, shared_mem_size>>>(
        rows, K,
        d_row_ptr,
        d_col_csr,
        d_val_f,
        d_X_f,
        d_Y_f);

    // Convert result back to bfloat16
    int total_Y = rows * K;
    int grid_convert_Y = (total_Y + block_convert - 1) / block_convert;
    convert_float_to_bf16_kernel_optimized<<<grid_convert_Y, block_convert>>>(
        d_Y_f, Y, total_Y);

    // Free intermediate buffers
    cudaFree(d_row_ptr);
    cudaFree(d_col_csr);
    cudaFree(d_val_f);
    cudaFree(d_X_f);
    cudaFree(d_Y_f);

    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void spmm_coo_float_kernel(
    int nnz,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* X,
    float* Y_float)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < nnz && k < K) {
        float val_f = __bfloat162float(values[idx]);
        int row = row_indices[idx];
        int col = col_indices[idx];

        float x_val_f = __bfloat162float(X[col * K + k]);
        float result_f = val_f * x_val_f;
        
        atomicAdd(&Y_float[row * K + k], result_f);
    }
}

__global__ void convert_float_to_bf16_kernel(float* float_array, __nv_bfloat16* bf16_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_array[idx] = __float2bfloat16(float_array[idx]);
    }
}

extern "C" void spmm_coo_bf16_origin(
    int rows,
    int nnz,
    int K,
    const __nv_bfloat16* values,
    const int* row_indices,
    const int* col_indices,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y) {
    
    float* d_Y_float;
    cudaMalloc(&d_Y_float, rows * K * sizeof(float));
    cudaMemset(d_Y_float, 0, rows * K * sizeof(float));
    
    dim3 block(16, 16, 1);
    dim3 grid((nnz + block.x - 1) / block.x, (K + block.y - 1) / block.y, 1);
    
    spmm_coo_float_kernel<<<grid, block>>>(nnz, K, values, row_indices, col_indices, X, d_Y_float);
    
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
    int* col_indices;
    int* row_indices;
    __nv_bfloat16* x;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> row_list = {1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16};
    int cols = 2048;
    int K = 4096;
    float density = 0.01f;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    
    for (int i = 0; i < row_list.size(); i++) {        
        TestCase test_case;
        test_case.rows = row_list[i];
        test_case.cols = cols;
        test_case.K = K;
        
        int total_elements = test_case.rows * test_case.cols;
        test_case.nnz = static_cast<int>(total_elements * density);

        if (test_case.nnz < test_case.rows) {
            test_case.nnz = test_case.rows;
        }
        
        test_case.values = new __nv_bfloat16[test_case.nnz];
        test_case.col_indices = new int[test_case.nnz];
        test_case.row_indices = new int[test_case.nnz];
        test_case.x = new __nv_bfloat16[test_case.cols * test_case.K];

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
        
        for (int j = 0; j < test_case.cols * test_case.K; j++) {
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
        const int value_item = test_case.nnz;
        const int row_index_item = test_case.nnz;
        const int col_index_item = test_case.nnz;
        const int x_item = test_case.cols * test_case.K;
        const int y_item = test_case.rows * test_case.K;
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
            spmm_coo_bf16_origin(test_case.rows, test_case.nnz, test_case.K, d_values, d_row_index, d_col_index, d_x, d_y);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            cudaMemset(d_y_optimized, 0, y_size);
            spmm_coo_bf16_optimized(test_case.rows, test_case.nnz, test_case.K, d_values, d_row_index, d_col_index, d_x, d_y_optimized);
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