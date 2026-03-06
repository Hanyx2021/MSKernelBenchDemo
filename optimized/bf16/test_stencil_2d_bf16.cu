#include <algorithm>
#include <cmath>
#include <cstdint>
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
// Configuration struct for launch parameters
struct Stencil2DBF16OptimizedConfig {
    dim3 block;
    dim3 grid;
};

// Shared-memory tiled 2D stencil kernel with fused boundary copy and vectorized loads (optimized)
__global__ void stencil_2d_bf16_kernel_optimized(
    __nv_bfloat16* u_new,
    const __nv_bfloat16* u_old,
    float r,
    int nx,
    int ny)
{
    extern __shared__ __nv_bfloat16 s_u[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global indices
    int j = bx * blockDim.x + tx;
    int i = by * blockDim.y + ty;

    // Dimensions of shared tile including halo
    int s_width = blockDim.x + 2;
    int s_height = blockDim.y + 2;
    int tile_elems = s_width * s_height;
    int tid = ty * blockDim.x + tx;
    int n_threads = blockDim.x * blockDim.y;

    // Phase 1: cooperative load into shared memory (with halo), vectorized bf16 loads
    const uint32_t* u_old_u32 = reinterpret_cast<const uint32_t*>(u_old);
    int pairs = tile_elems / 2;
    for (int pid = tid; pid < pairs; pid += n_threads) {
        int idx = pid * 2;            // shared-memory flat index for first of two
        int si  = idx / s_width;
        int sj0 = idx % s_width;
        int sj1 = sj0 + 1;
        int gi  = by * blockDim.y + (si - 1);
        int gj0 = bx * blockDim.x + (sj0 - 1);
        int gj1 = bx * blockDim.x + (sj1 - 1);
        int base_id = gi * ny + gj0;
        // both elements in-range for vector load
        if (gi >= 0 && gi < nx && gj0 >= 0 && gj1 < ny) {
            int vec_idx = base_id >> 1; // index of packed bf16 pair
            uint32_t packed = u_old_u32[vec_idx];
            union {
                uint32_t u32;
                __nv_bfloat16 v16[2];
            } tmp;
            tmp.u32 = packed;
            int sel0 = base_id & 1;
            s_u[idx]     = tmp.v16[sel0];
            s_u[idx + 1] = tmp.v16[sel0 ^ 1];
        } else {
            // fallback to scalar loads
            __nv_bfloat16 v0 = (gi >= 0 && gi < nx && gj0 >= 0 && gj0 < ny)
                               ? u_old[base_id]
                               : __float2bfloat16(0.0f);
            __nv_bfloat16 v1 = (gi >= 0 && gi < nx && gj1 >= 0 && gj1 < ny)
                               ? u_old[base_id + 1]
                               : __float2bfloat16(0.0f);
            s_u[idx]     = v0;
            s_u[idx + 1] = v1;
        }
    }
    __syncthreads();

    // Phase 2: compute stencil or copy boundary
    if (i < nx && j < ny) {
        float result;
        int gid = i * ny + j;
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            // boundary condition: direct copy
            result = __bfloat162float(u_old[gid]);
        } else {
            int si = ty + 1;
            int sj = tx + 1;
            int lidx = si * s_width + sj;
            float center = __bfloat162float(s_u[lidx]);
            float left   = __bfloat162float(s_u[lidx - 1]);
            float right  = __bfloat162float(s_u[lidx + 1]);
            float up     = __bfloat162float(s_u[lidx + s_width]);
            float down   = __bfloat162float(s_u[lidx - s_width]);
            result = center + r * (left + right + up + down - 4.0f * center);
        }
        u_new[gid] = __float2bfloat16(result);
    }
}

// External C wrapper using the optimized kernel
extern "C" void stencil_2d_bf16_optimized(
    __nv_bfloat16* u_new,
    const __nv_bfloat16* u_old,
    float r,
    int nx,
    int ny)
{
    // Launch parameters: blockDim.x must be multiple of 32 for coalescing
    Stencil2DBF16OptimizedConfig cfg;
    cfg.block = dim3(32, 8);
    cfg.grid  = dim3((ny + cfg.block.x - 1) / cfg.block.x,
                     (nx + cfg.block.y - 1) / cfg.block.y);
    // Shared memory size: (blockDim.x+2)*(blockDim.y+2) elements
    size_t shared_bytes = (cfg.block.x + 2) * (cfg.block.y + 2) * sizeof(__nv_bfloat16);

    stencil_2d_bf16_kernel_optimized<<<cfg.grid, cfg.block, shared_bytes>>>(
        u_new, u_old, r, nx, ny);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====

__global__ void stencil_2d_bf16_kernel(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1) {
        int idx = i * ny + j;

        float center = __bfloat162float(u_old[idx]);
        float left = __bfloat162float(u_old[idx - 1]);
        float right = __bfloat162float(u_old[idx + 1]);
        float up = __bfloat162float(u_old[idx + ny]);
        float down = __bfloat162float(u_old[idx - ny]);
        
        u_new[idx] = __float2bfloat16(center + r * (left + right + up + down - 4.0f * center));
    }
}

__global__ void copy_boundary_bf16_kernel(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            dst[i * ny + j] = src[i * ny + j];
        }
    }
}

extern "C" void stencil_2d_bf16_origin(
    __nv_bfloat16* u_new, 
    const __nv_bfloat16* u_old, 
    float r, int nx, int ny) {
    
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);
    
    stencil_2d_bf16_kernel<<<grid, block>>>(u_new, u_old, r, nx, ny);

    copy_boundary_bf16_kernel<<<grid, block>>>(u_new, u_old, nx, ny);

    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int nx;
    int ny;
    __nv_bfloat16 *u_old;
} TestCase;

// Function to load test case from hardcoded values

void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> nx_list = {1 << 10, 1 << 12, 1 << 14};
    std::vector<int> ny_list = {1 << 10, 1 << 12, 1 << 14};

    for (int i = 0; i < nx_list.size(); i++) 
        for(int j = 0; j < ny_list.size(); j++) {
        TestCase test_case;
        test_case.nx = nx_list[i];
        test_case.ny = ny_list[j];
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Random seed for testing
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int u_item = test_case.nx * test_case.ny;
        test_case.u_old = new __nv_bfloat16[u_item];
        
        for (int ii = 0; ii < u_item; ii++) {
            test_case.u_old[ii] = input_dist(rng);
        }
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: nx: %d, ny: %d. Complexity: %d\n", test_case.nx, test_case.ny, test_case.nx * test_case.ny);
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
        const int u_item = test_case.nx * test_case.ny;
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
            stencil_2d_bf16_origin(d_u_old, d_u_new, 0.25f, test_case.nx, test_case.ny);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        
        auto optimized_kernel = [&]() {
            stencil_2d_bf16_optimized(d_u_old, d_u_new_optimized, 0.25f, test_case.nx, test_case.ny);
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