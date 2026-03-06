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

__global__ void stencil_3d_bf16_kernel_optimized(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny, int nz) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= 1 && i < nx-1 && 
        j >= 1 && j < ny-1 && 
        k >= 1 && k < nz-1) {
        
        int idx = i * ny * nz + j * nz + k;
        
        float center = __bfloat162float(u_old[idx]);
        
        float left   = __bfloat162float(u_old[idx - ny*nz]);
        float right  = __bfloat162float(u_old[idx + ny*nz]);

        float up     = __bfloat162float(u_old[idx + nz]);
        float down   = __bfloat162float(u_old[idx - nz]);
        
        float front  = __bfloat162float(u_old[idx + 1]);
        float back   = __bfloat162float(u_old[idx - 1]);

        u_new[idx] = __float2bfloat16(center + r * (left + right + up + down + front + back - 6.0f * center));
    }
}

__global__ void copy_boundary_3d_bf16_kernel_optimized(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    int nx, int ny, int nz) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz) {
        bool is_boundary = 
            (i == 0) || (i == nx - 1) ||
            (j == 0) || (j == ny - 1) ||
            (k == 0) || (k == nz - 1);
        
        if (is_boundary) {
            int idx = i * ny * nz + j * nz + k;
            dst[idx] = src[idx];
        }
    }
}

extern "C" void stencil_3d_bf16_optimized(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny, int nz) {
    
    dim3 block(4, 4, 4);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );
    
    stencil_3d_bf16_kernel_optimized<<<grid, block>>>(u_new, u_old, r, nx, ny, nz);

    copy_boundary_3d_bf16_kernel_optimized<<<grid, block>>>(u_new, u_old, nx, ny, nz);
    
    cudaDeviceSynchronize();
}

// ==== OPTIMIZED KERNEL END ====

__global__ void stencil_3d_bf16_kernel(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny, int nz) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= 1 && i < nx-1 && 
        j >= 1 && j < ny-1 && 
        k >= 1 && k < nz-1) {
        
        int idx = i * ny * nz + j * nz + k;
        
        float center = __bfloat162float(u_old[idx]);
        
        float left   = __bfloat162float(u_old[idx - ny*nz]);
        float right  = __bfloat162float(u_old[idx + ny*nz]);

        float up     = __bfloat162float(u_old[idx + nz]);
        float down   = __bfloat162float(u_old[idx - nz]);
        
        float front  = __bfloat162float(u_old[idx + 1]);
        float back   = __bfloat162float(u_old[idx - 1]);

        u_new[idx] = __float2bfloat16(center + r * (left + right + up + down + front + back - 6.0f * center));
    }
}

__global__ void copy_boundary_3d_bf16_kernel(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    int nx, int ny, int nz) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz) {
        bool is_boundary = 
            (i == 0) || (i == nx - 1) ||
            (j == 0) || (j == ny - 1) ||
            (k == 0) || (k == nz - 1);
        
        if (is_boundary) {
            int idx = i * ny * nz + j * nz + k;
            dst[idx] = src[idx];
        }
    }
}

extern "C" void stencil_3d_bf16_origin(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny, int nz) {
    
    dim3 block(4, 4, 4);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );
    
    stencil_3d_bf16_kernel<<<grid, block>>>(u_new, u_old, r, nx, ny, nz);

    copy_boundary_3d_bf16_kernel<<<grid, block>>>(u_new, u_old, nx, ny, nz);
    
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int nx;
    int ny;
    int nz;
    __nv_bfloat16 *u_old;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> nx_list = {1 << 6, 1 << 7, 1 << 8};
    std::vector<int> ny_list = {1 << 6, 1 << 7, 1 << 8};
    std::vector<int> nz_list = {1 << 6, 1 << 7, 1 << 8};

    for (int i = 0; i < nx_list.size(); i++) 
        for(int j = 0; j < ny_list.size(); j++) 
            for(int k = 0; k < nz_list.size(); k++)
            {
                TestCase test_case;
                test_case.nx = nx_list[i];
                test_case.ny = ny_list[j];
                test_case.nz = nz_list[k];
                
                // Use fixed seed for reproducibility
                std::random_device rd;
                std::mt19937 rng(rd());  // Random seed for testing
                std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
                
                int u_item = test_case.nx * test_case.ny * test_case.nz;
                test_case.u_old = new __nv_bfloat16[u_item];
                
                for (int ii = 0; ii < u_item; ii++) {
                    test_case.u_old[ii] = input_dist(rng);
                }
                test_case_list.push_back(test_case);
            }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: nx: %d, ny: %d, nz: %d. Complexity: %ld\n", test_case.nx, test_case.ny, test_case.nz, (long)test_case.nx * test_case.ny * test_case.nz);
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
        const int u_item = test_case.nx * test_case.ny * test_case.nz;
        size_t u_size = u_item * sizeof(__nv_bfloat16);

        // Host memory inputs
        __nv_bfloat16* h_u_old = (__nv_bfloat16*)malloc(u_size);
        __nv_bfloat16* h_u_new = (__nv_bfloat16*)malloc(u_size);
        __nv_bfloat16* h_u_new_optimized = (__nv_bfloat16*)malloc(u_size);

        if (!h_u_old || !h_u_new || !h_u_new_optimized) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_u_old, test_case.u_old, u_size);

        // GPU memory allocation
        __nv_bfloat16 *d_u_old, *d_u_new, *d_u_new_optimized;

        checkCudaError(cudaMalloc((void**)&d_u_old, u_size), "Allocating d_u_old");
        checkCudaError(cudaMalloc((void**)&d_u_new, u_size), "Allocating d_u_new");
        checkCudaError(cudaMalloc((void**)&d_u_new_optimized, u_size), "Allocating d_u_new_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_u_old, h_u_old, u_size, cudaMemcpyHostToDevice), "Copying h_u_old to d_u_old");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;  // Reduced iterations for stability

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            stencil_3d_bf16_origin(d_u_old, d_u_new, 0.25f, test_case.nx, test_case.ny, test_case.nz);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            stencil_3d_bf16_optimized(d_u_old, d_u_new_optimized, 0.25f, test_case.nx, test_case.ny, test_case.nz);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(h_u_new, d_u_new, u_size, cudaMemcpyDeviceToHost), "Copying d_u_new to h_u_new");
        checkCudaError(cudaMemcpy(h_u_new_optimized, d_u_new_optimized, u_size, cudaMemcpyDeviceToHost), "Copying d_u_new_optimized to h_u_new_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        for (int i = 0; i < u_item; i++) {
            if (!bfloat16_equals(h_u_new[i], h_u_new_optimized[i], 1e-2f)) {
                printf("Output mismatch at index %d: original %.6f, optimized %.6f\n", i, __bfloat162float(h_u_new[i]), __bfloat162float(h_u_new_optimized[i]));
                return 1;
            }
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_u_old);
        cudaFree(d_u_new);
        cudaFree(d_u_new_optimized);

        free(h_u_old);
        free(h_u_new);
        free(h_u_new_optimized);
        delete [] test_case.u_old;
    }

    return 0;
}