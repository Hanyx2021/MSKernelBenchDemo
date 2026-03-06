#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

#define MAX_TWIDDLE_PER_BLOCK 1024
#define PI 3.14159265358979323846f
// Number of initial stages to fuse into one block FFT
#define FUSE_STAGES 8
#define MAX_FUSE_TILE_SIZE (1 << FUSE_STAGES)

// Bit-reversal permutation kernel
__global__ void bit_reverse_permute_kernel_optimized(
    float* out_real, float* out_img,
    const float* in_real, const float* in_img,
    int N, int log2N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    unsigned int rev = __brev((unsigned int)idx) >> (32 - log2N);
    out_real[idx] = in_real[rev];
    out_img[idx]  = in_img[rev];
}

// In-place block FFT kernel fusing first L stages into shared memory
__global__ void block_fft_kernel_optimized(
    float* data_real, float* data_img,
    int N, int fuse_stages) {
    // Shared memory buffers
    __shared__ float s_real[MAX_FUSE_TILE_SIZE];
    __shared__ float s_img[MAX_FUSE_TILE_SIZE];

    int tile_size = 1 << fuse_stages;
    int half_tile = tile_size >> 1;
    int tid = threadIdx.x;
    int base = blockIdx.x * tile_size;

    // Load data into shared memory (two elements per thread)
    if (tid < half_tile) {
        int idx0 = base + tid;
        int idx1 = base + tid + half_tile;
        s_real[tid] = data_real[idx0];
        s_img[tid]  = data_img[idx0];
        s_real[tid + half_tile] = data_real[idx1];
        s_img[tid + half_tile]  = data_img[idx1];
    }
    __syncthreads();

    // Perform fused FFT stages entirely in shared memory
    for (int stage = 1; stage <= fuse_stages; ++stage) {
        int butterfly_size = 1 << stage;
        int half_bfly      = butterfly_size >> 1;
        int local_tid = tid;
        // Only threads < tile_size/2 participate; extra threads idle
        if (local_tid < (tile_size >> 1)) {
            int group_id = local_tid >> (stage - 1);
            int pair_id  = local_tid & (half_bfly - 1);
            int idx1     = group_id * butterfly_size + pair_id;
            int idx2     = idx1 + half_bfly;

            // Load values
            float x1r = s_real[idx1];
            float x1i = s_img[idx1];
            float x2r = s_real[idx2];
            float x2i = s_img[idx2];

            // Compute twiddle factor: exp(-2*pi*i*pair_id/butterfly_size)
            float twr, twi;
            if (pair_id == 0) {
                twr = 1.0f; twi = 0.0f;
            } else {
                float angle = -2.0f * PI * pair_id / (float)butterfly_size;
                twr = cosf(angle);
                twi = sinf(angle);
            }
            // Butterfly computation
            float y_r = twr * x2r - twi * x2i;
            float y_i = twr * x2i + twi * x2r;
            s_real[idx1] = x1r + y_r;
            s_img[idx1]  = x1i + y_i;
            s_real[idx2] = x1r - y_r;
            s_img[idx2]  = x1i - y_i;
        }
        __syncthreads();
    }

    // Write results back to global memory
    if (tid < half_tile) {
        int idx0 = base + tid;
        int idx1 = base + tid + half_tile;
        data_real[idx0] = s_real[tid];
        data_img[idx0]  = s_img[tid];
        data_real[idx1] = s_real[tid + half_tile];
        data_img[idx1]  = s_img[tid + half_tile];
    }
}

// Butterfly operation kernel for remaining stages
__global__ void butterfly_kernel_optimized(
    float* data_real, float* data_img,
    const float* twiddle_real, const float* twiddle_img,
    int N, int stage) {
    int butterfly_size = 1 << stage;
    int half_bfly      = butterfly_size >> 1;
    int idx            = blockIdx.x * blockDim.x + threadIdx.x;
    int num_butterflies= N >> 1;
    if (idx >= num_butterflies) return;

    // Shared staging of twiddle
    __shared__ float sh_tw_real[MAX_TWIDDLE_PER_BLOCK];
    __shared__ float sh_tw_img[MAX_TWIDDLE_PER_BLOCK];

    int stride  = N >> stage;
    int num_tw  = (half_bfly < blockDim.x ? half_bfly : blockDim.x);
    if (threadIdx.x < num_tw) {
        sh_tw_real[threadIdx.x] = twiddle_real[threadIdx.x * stride];
        sh_tw_img[threadIdx.x]  = twiddle_img[threadIdx.x * stride];
    }
    __syncthreads();

    int group_id = idx / half_bfly;
    int pair_id  = idx % half_bfly;
    int idx1     = group_id * butterfly_size + pair_id;
    int idx2     = idx1 + half_bfly;

    float x1r = data_real[idx1];
    float x1i = data_img[idx1];
    float x2r = data_real[idx2];
    float x2i = data_img[idx2];

    float twr, twi;
    if (pair_id < num_tw) {
        twr = sh_tw_real[pair_id];
        twi = sh_tw_img[pair_id];
    } else {
        twr = twiddle_real[pair_id * stride];
        twi = twiddle_img[pair_id * stride];
    }

    float y_r = twr * x2r - twi * x2i;
    float y_i = twr * x2i + twi * x2r;

    data_real[idx1] = x1r + y_r;
    data_img[idx1]  = x1i + y_i;
    data_real[idx2] = x1r - y_r;
    data_img[idx2]  = x1i - y_i;
}

// Host helper to compute twiddle factors for full-size FFT
static void compute_twiddle_factors_optimized(float* real, float* img, int N) {
    for (int i = 0; i < N/2; ++i) {
        float angle = -2.0f * PI * i / (float)N;
        real[i] = cosf(angle);
        img[i]  = sinf(angle);
    }
}

extern "C" void FFT_optimized(
    const float* input_real,
    const float* input_img,
    float* output_real,
    float* output_img,
    int N) {
    // Device buffers
    float *d_input_real, *d_input_img;
    float *d_work_real, *d_work_img;
    float *d_twiddle_real, *d_twiddle_img;
    size_t mem_size = N * sizeof(float);

    cudaMalloc(&d_input_real,  mem_size);
    cudaMalloc(&d_input_img,   mem_size);
    cudaMalloc(&d_work_real,   mem_size);
    cudaMalloc(&d_work_img,    mem_size);

    cudaMemcpy(d_input_real, input_real, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_img,  input_img,  mem_size, cudaMemcpyHostToDevice);

    // Prepare twiddle factors
    float* h_tw_real = (float*)malloc((N/2) * sizeof(float));
    float* h_tw_img  = (float*)malloc((N/2) * sizeof(float));
    compute_twiddle_factors_optimized(h_tw_real, h_tw_img, N);
    cudaMalloc(&d_twiddle_real, (N/2) * sizeof(float));
    cudaMalloc(&d_twiddle_img,  (N/2) * sizeof(float));
    cudaMemcpy(d_twiddle_real, h_tw_real, (N/2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddle_img,  h_tw_img,  (N/2) * sizeof(float), cudaMemcpyHostToDevice);

    // Compute log2(N)
    int log2N = 0;
    for (int temp = N; temp > 1; temp >>= 1) ++log2N;

    // Bit-reversal permutation
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    bit_reverse_permute_kernel_optimized<<<blocks, threads_per_block>>>(
        d_work_real, d_work_img,
        d_input_real, d_input_img,
        N, log2N);
    cudaDeviceSynchronize();

    // Fused block FFT for initial stages
    int fuse_stages = (log2N < FUSE_STAGES ? log2N : FUSE_STAGES);
    if (fuse_stages > 0) {
        int tile_size = 1 << fuse_stages;
        int half_tile = tile_size >> 1;
        int blocks_fuse = N / tile_size;
        // Launch with static block size = MAX_FUSE_TILE_SIZE/2 threads
        block_fft_kernel_optimized<<<blocks_fuse, MAX_FUSE_TILE_SIZE/2>>>(
            d_work_real, d_work_img,
            N, fuse_stages);
        cudaDeviceSynchronize();
    }

    // Remaining FFT stages
    for (int stage = fuse_stages + 1; stage <= log2N; ++stage) {
        int num_butterflies = N >> 1;
        int blocks_bfly = (num_butterflies + threads_per_block - 1) / threads_per_block;
        butterfly_kernel_optimized<<<blocks_bfly, threads_per_block>>>(
            d_work_real, d_work_img,
            d_twiddle_real, d_twiddle_img,
            N, stage);
        cudaDeviceSynchronize();
    }

    // Copy result back to host
    cudaMemcpy(output_real, d_work_real, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_img,  d_work_img,  mem_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input_real);
    cudaFree(d_input_img);
    cudaFree(d_work_real);
    cudaFree(d_work_img);
    cudaFree(d_twiddle_real);
    cudaFree(d_twiddle_img);
    free(h_tw_real);
    free(h_tw_img);
}
