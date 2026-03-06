#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

#define MAX_TWIDDLE (1<<13)

// Twiddle factors in constant memory
__constant__ __nv_bfloat16 d_twiddle_real_const[MAX_TWIDDLE];
__constant__ __nv_bfloat16 d_twiddle_img_const[MAX_TWIDDLE];

// Bit-reversal permutation kernel
__global__ void bit_reverse_permute_bf16_kernel_optimized(
    __nv_bfloat16* out_real, __nv_bfloat16* out_img,
    const __nv_bfloat16* in_real, const __nv_bfloat16* in_img,
    int N, int logN)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    unsigned int rev = __brev((unsigned int)idx) >> (32 - logN);
    out_real[idx] = in_real[rev];
    out_img[idx]  = in_img[rev];
}

// Butterfly kernel for all stages
__global__ void butterfly_bf16_kernel_optimized(
    __nv_bfloat16* data_real, __nv_bfloat16* data_img,
    int N, int stage)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_butterflies = N >> 1;
    if (idx >= num_butterflies) return;

    int butterfly_size = 1 << stage;
    int half_size = butterfly_size >> 1;
    int group_id = idx / half_size;
    int pair_id = idx % half_size;
    int idx1 = group_id * butterfly_size + pair_id;
    int idx2 = idx1 + half_size;

    float x1_real = __bfloat162float(data_real[idx1]);
    float x1_img  = __bfloat162float(data_img[idx1]);
    float x2_real = __bfloat162float(data_real[idx2]);
    float x2_img  = __bfloat162float(data_img[idx2]);

    int twiddle_idx = pair_id * (N >> stage);
    float tw_real = __bfloat162float(d_twiddle_real_const[twiddle_idx]);
    float tw_img  = __bfloat162float(d_twiddle_img_const[twiddle_idx]);

    float y_real = tw_real * x2_real - tw_img * x2_img;
    float y_img  = tw_real * x2_img  + tw_img * x2_real;

    data_real[idx1] = __float2bfloat16(x1_real + y_real);
    data_img[idx1]  = __float2bfloat16(x1_img  + y_img);
    data_real[idx2] = __float2bfloat16(x1_real - y_real);
    data_img[idx2]  = __float2bfloat16(x1_img  - y_img);
}

extern "C" void FFT_bf16_optimized(
    const __nv_bfloat16* input_real,
    const __nv_bfloat16* input_img,
    __nv_bfloat16*       output_real,
    __nv_bfloat16*       output_img,
    int                  N)
{
    // compute log2(N)
    int logN = 0;
    for (int t = N; t > 1; t >>= 1) ++logN;

    // build twiddles on host and copy to constant memory
    int half = N >> 1;
    __nv_bfloat16* h_tw_r = new __nv_bfloat16[half];
    __nv_bfloat16* h_tw_i = new __nv_bfloat16[half];
    for (int i = 0; i < half; ++i) {
        float angle = -2.0f * M_PI * i / float(N);
        h_tw_r[i] = __float2bfloat16(cosf(angle));
        h_tw_i[i] = __float2bfloat16(sinf(angle));
    }
    cudaMemcpyToSymbol(d_twiddle_real_const, h_tw_r, half * sizeof(__nv_bfloat16));
    cudaMemcpyToSymbol(d_twiddle_img_const,  h_tw_i, half * sizeof(__nv_bfloat16));
    delete[] h_tw_r;
    delete[] h_tw_i;

    // 1) bit-reverse permutation
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    bit_reverse_permute_bf16_kernel_optimized<<<blocks, threads>>>(
        output_real, output_img,
        input_real,  input_img,
        N, logN);
    cudaDeviceSynchronize();

    // 2) radix-2 butterfly for all stages
    for (int stage = 1; stage <= logN; ++stage) {
        int nb = N >> 1;
        blocks = (nb + threads - 1) / threads;
        butterfly_bf16_kernel_optimized<<<blocks, threads>>>(
            output_real, output_img,
            N, stage);
        cudaDeviceSynchronize();
    }
}