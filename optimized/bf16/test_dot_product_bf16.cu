#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <functional>
#include <math.h>
#include <optional>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>


// Simple function to check if two floats are approximately equal
bool float_equals_relative(float a, float b, float tolerance) {
    if (a == b) return true;
    if (a == 0.0f || b == 0.0f) {
        return fabs(a - b) < tolerance;
    }
    return fabs(a - b) / fmax(fabs(a), fabs(b)) < tolerance;
}

// ==== OPTIMIZED KERNEL START ====
// Optimized BF16 dot product kernel with higher unroll (8 elements) and software pipelining
static const int WARP_SIZE_OPTIM = 32;
static const int MAX_WARPS_OPTIM = 32;  // supports up to 1024 threads per block

// Union to unpack four bf16 values from a 64-bit packed word
union BF16x4Optimized {
    uint64_t packed;
    __nv_bfloat16 v[4];
};

__global__ void dot_product_bf16_kernel_optimized(
    float*            __restrict__ loss,
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ Y,
    const int             N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    // Check 8-byte alignment for vectorized bf16 loads
    bool aligned8 = (((uintptr_t)X & 7) == 0) && (((uintptr_t)Y & 7) == 0);

    if (aligned8) {
        const uint64_t* Xp64 = reinterpret_cast<const uint64_t*>(X);
        const uint64_t* Yp64 = reinterpret_cast<const uint64_t*>(Y);
        unsigned int groupCount = N >> 3;       // number of complete 8-element groups
        unsigned int stepG = stride;

        // Software pipelined prefetch for two groups
        unsigned int g0 = tid;
        unsigned int g1 = g0 + stepG;
        uint64_t x0_lo = 0, x0_hi = 0, y0_lo = 0, y0_hi = 0;
        uint64_t x1_lo = 0, x1_hi = 0, y1_lo = 0, y1_hi = 0;
        if (g0 < groupCount) {
            unsigned int base0 = g0 * 2;
            x0_lo = __ldg(Xp64 + base0);
            x0_hi = __ldg(Xp64 + base0 + 1);
            y0_lo = __ldg(Yp64 + base0);
            y0_hi = __ldg(Yp64 + base0 + 1);
        }
        if (g1 < groupCount) {
            unsigned int base1 = g1 * 2;
            x1_lo = __ldg(Xp64 + base1);
            x1_hi = __ldg(Xp64 + base1 + 1);
            y1_lo = __ldg(Yp64 + base1);
            y1_hi = __ldg(Yp64 + base1 + 1);
        }

        // Main pipelined loop
        while (g0 < groupCount) {
            // Convert & multiply group g0
            BF16x4Optimized ux_lo{ x0_lo };
            BF16x4Optimized ux_hi{ x0_hi };
            BF16x4Optimized uy_lo{ y0_lo };
            BF16x4Optimized uy_hi{ y0_hi };
            float fx0 = __bfloat162float(ux_lo.v[0]);
            float fx1 = __bfloat162float(ux_lo.v[1]);
            float fx2 = __bfloat162float(ux_lo.v[2]);
            float fx3 = __bfloat162float(ux_lo.v[3]);
            float fx4 = __bfloat162float(ux_hi.v[0]);
            float fx5 = __bfloat162float(ux_hi.v[1]);
            float fx6 = __bfloat162float(ux_hi.v[2]);
            float fx7 = __bfloat162float(ux_hi.v[3]);
            float fy0 = __bfloat162float(uy_lo.v[0]);
            float fy1 = __bfloat162float(uy_lo.v[1]);
            float fy2 = __bfloat162float(uy_lo.v[2]);
            float fy3 = __bfloat162float(uy_lo.v[3]);
            float fy4 = __bfloat162float(uy_hi.v[0]);
            float fy5 = __bfloat162float(uy_hi.v[1]);
            float fy6 = __bfloat162float(uy_hi.v[2]);
            float fy7 = __bfloat162float(uy_hi.v[3]);
            sum += fx0*fy0 + fx1*fy1 + fx2*fy2 + fx3*fy3
                 + fx4*fy4 + fx5*fy5 + fx6*fy6 + fx7*fy7;

            // Advance pipeline
            g0 = g1;
            x0_lo = x1_lo; x0_hi = x1_hi; y0_lo = y1_lo; y0_hi = y1_hi;
            g1 += stepG;
            if (g1 < groupCount) {
                unsigned int base1n = g1 * 2;
                x1_lo = __ldg(Xp64 + base1n);
                x1_hi = __ldg(Xp64 + base1n + 1);
                y1_lo = __ldg(Yp64 + base1n);
                y1_hi = __ldg(Yp64 + base1n + 1);
            }
        }

        // Cleanup leftover elements (<8 per thread) using scalar loads
        unsigned int processed = groupCount * 8;
        for (unsigned int j = processed + tid; j < (unsigned int)N; j += stride) {
            float xf = __bfloat162float(__ldg(X + j));
            float yf = __bfloat162float(__ldg(Y + j));
            sum += xf * yf;
        }
    } else {
        // Fallback to scalar/vector loads (unaligned)
        for (unsigned int i = tid * 2; i + 1 < (unsigned int)N; i += stride * 2) {
            float x0 = __bfloat162float(__ldg(X + i));
            float x1 = __bfloat162float(__ldg(X + i + 1));
            float y0 = __bfloat162float(__ldg(Y + i));
            float y1 = __bfloat162float(__ldg(Y + i + 1));
            sum += x0 * y0 + x1 * y1;
        }
        if ((N & 1) && tid == (unsigned int)(N - 1)) {
            float xt = __bfloat162float(__ldg(X + tid));
            float yt = __bfloat162float(__ldg(Y + tid));
            sum += xt * yt;
        }
    }

    // Intra-warp reduction using shuffle down
    unsigned int lane   = threadIdx.x & (WARP_SIZE_OPTIM - 1);
    unsigned int warpId = threadIdx.x >> 5;
    unsigned int fullMask = 0xffffffffu;
    for (int offset = WARP_SIZE_OPTIM / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(fullMask, sum, offset);
    }

    // Shared memory for warp sums
    __shared__ float warp_sums[MAX_WARPS_OPTIM];
    if (lane == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    float block_sum = 0.0f;
    if (threadIdx.x < WARP_SIZE_OPTIM) {
        block_sum = warp_sums[lane];
    }
    for (int offset = WARP_SIZE_OPTIM / 2; offset > 0; offset >>= 1) {
        block_sum += __shfl_down_sync(fullMask, block_sum, offset);
    }

    // Single atomic update per block
    if (lane == 0) {
        atomicAdd(loss, block_sum);
    }
}

extern "C" void dot_product_bf16_optimized(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {
    // Tuned block size for occupancy under higher unroll/register pressure
    const int threadsPerBlock = 256;
    int groupCount = N >> 3;
    int blocks = (groupCount + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;

    dot_product_bf16_kernel_optimized<<<blocks, threadsPerBlock>>>(loss, X, Y, N);
    cudaDeviceSynchronize();
}
// ==== OPTIMIZED KERNEL END ====
 
__global__ void dot_product_bf16_kernel(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {
    
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    
    for (int i = tid; i < N; i += stride) {
        float x_val = __bfloat162float(X[i]);
        float y_val = __bfloat162float(Y[i]);
        
        sum += x_val * y_val;
    }
    
    if (sum != 0.0f) {
        atomicAdd(loss, sum);
    }
}

extern "C" void dot_product_bf16_origin(
    float* loss,
    const __nv_bfloat16* X,
    const __nv_bfloat16* Y,
    const int N) {
    
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;
    
    dot_product_bf16_kernel<<<blocks, threadsPerBlock>>>(loss, X, Y, N);
    cudaDeviceSynchronize();
}

// Test case input data structure
typedef struct {
    int N;
    __nv_bfloat16 *X;
    __nv_bfloat16 *Y;
} TestCase;

// Function to load test case from hardcoded values
void load_test_case(std::vector<TestCase>& test_case_list) {
    std::vector<int> N_list = {1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20};

    for (int i = 0; i < N_list.size(); i++) {
        TestCase test_case;
        test_case.N = N_list[i];
        
        // Use fixed seed for reproducibility
        std::random_device rd;
        std::mt19937 rng(rd());  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        
        int item_count = test_case.N;
        test_case.X = new __nv_bfloat16[item_count];
        test_case.Y = new __nv_bfloat16[item_count];
        
        for (int ii = 0; ii < item_count; ii++) {
            test_case.X[ii] = __float2bfloat16(input_dist(rng));
        }
        for (int ii = 0; ii < item_count; ii++) {
            test_case.Y[ii] = __float2bfloat16(input_dist(rng));
        }
        test_case_list.push_back(test_case);
    }
}

// Print test case data size
void print_test_case_size(TestCase test_case) {
    printf("Test case size: N: %d. Complexity: %d\n", test_case.N, test_case.N);
}

// Function to warm up GPU and stabilize frequency
void stabilize_gpu() {
    // Create a dummy kernel to warm up GPU
    __nv_bfloat16 *d_temp;
    cudaMalloc(&d_temp, sizeof(__nv_bfloat16));
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
    
    return total_time;  // Total time for all iterations
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
        const int item_count = test_case.N;
        size_t data_size = item_count * sizeof(__nv_bfloat16);
        size_t loss_size = sizeof(__nv_bfloat16) * (test_case.N / 1024 + 1); // Extra space for optimized kernel

        // Host memory inputs
        __nv_bfloat16* h_X = (__nv_bfloat16*)malloc(data_size);
        __nv_bfloat16* h_Y = (__nv_bfloat16*)malloc(data_size);
        float h_loss, h_loss_optimized;

        if (!h_X || !h_Y) {
            printf("Failed to allocate host memory\n");
            return 1;
        }

        // Copy data to host
        memcpy(h_X, test_case.X, data_size);
        memcpy(h_Y, test_case.Y, data_size);

        // GPU memory allocation
        __nv_bfloat16 *d_X, *d_Y;
        float *d_loss, *d_loss_optimized;

        checkCudaError(cudaMalloc((void**)&d_X, data_size), "Allocating d_X");
        checkCudaError(cudaMalloc((void**)&d_Y, data_size), "Allocating d_Y");
        checkCudaError(cudaMalloc((void**)&d_loss, loss_size), "Allocating d_loss");
        checkCudaError(cudaMalloc((void**)&d_loss_optimized, loss_size), "Allocating d_loss_optimized");

        // Copy input data to GPU
        checkCudaError(cudaMemcpy(d_X, h_X, data_size, cudaMemcpyHostToDevice), "Copying h_X to d_X");
        checkCudaError(cudaMemcpy(d_Y, h_Y, data_size, cudaMemcpyHostToDevice), "Copying h_Y to d_Y");

        // Stabilize GPU frequency
        stabilize_gpu();
        const char* profiling_env = std::getenv("PROFILING_MODE");
        const int ITERATIONS = profiling_env ? 1 : 50;

        /* ================  Define test kernels  ================ */

        auto origin_kernel = [&]() {
            float zero = 0.0f;
            checkCudaError(cudaMemcpy(d_loss, &zero, sizeof(float), cudaMemcpyHostToDevice), "Initializing d_loss");
            dot_product_bf16_origin(d_loss, d_X, d_Y, test_case.N);
            checkCudaError(cudaGetLastError(), "Origin kernel launch");
        };
        auto optimized_kernel = [&]() {
            float zero = 0.0f;
            checkCudaError(cudaMemcpy(d_loss_optimized, &zero, sizeof(float), cudaMemcpyHostToDevice), "Initializing d_loss_optimized");
            dot_product_bf16_optimized(d_loss_optimized, d_X, d_Y, test_case.N);
            checkCudaError(cudaGetLastError(), "Optimized kernel launch");
        };

        /* ================ Run test kernels  ================ */
        float origin_time = measure_kernel_performance(origin_kernel, ITERATIONS);
        
        stabilize_gpu();
        
        float optimized_time = measure_kernel_performance(optimized_kernel, ITERATIONS);

        // Copy results back for verification
        checkCudaError(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost), "Copying d_loss to h_loss");
        checkCudaError(cudaMemcpy(&h_loss_optimized, d_loss_optimized, sizeof(float), cudaMemcpyDeviceToHost), "Copying d_loss_optimized to h_loss_optimized");

        /* ================  Verify results  ================ */
        printf("===================\n");
        print_test_case_size(test_case);

        if (!float_equals_relative(h_loss, h_loss_optimized, 1e-3f)) {
            printf("Output mismatch: original %.6f, optimized %.6f\n", h_loss, h_loss_optimized);
            return 1;
        }

        /* ================  Calculate performance  ================ */
        printf("Speedup ratio: %.2f\n", origin_time / optimized_time);

        /* ================  Cleanup  ================ */
        cudaFree(d_X);
        cudaFree(d_Y);
        cudaFree(d_loss);
        cudaFree(d_loss_optimized);

        free(h_X);
        free(h_Y);
        delete [] test_case.X;
        delete [] test_case.Y;
    }

    return 0;
}