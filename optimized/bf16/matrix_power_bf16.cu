#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Scalar BF16 matrix multiplication kernel (optimized)
__global__ void matrix_mul_bf16_scalar_kernel_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            float a_val = __bfloat162float(A[row * N + k]);
            float b_val = __bfloat162float(B[k * N + col]);
            sum += a_val * b_val;
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

// External C wrapper for matrix power (optimized – now using the same
// repeated-multiply strategy as the origin version so that rounding
// is identical)
extern "C" void matrix_power_bf16_optimized(
    const __nv_bfloat16* A,
    __nv_bfloat16* B,
    int N,
    int P) {
    size_t bytes = size_t(N) * N * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_base = nullptr, *d_result = nullptr, *d_temp = nullptr;

    // Allocate device buffers
    cudaMalloc(&d_base, bytes);
    cudaMalloc(&d_result, bytes);
    cudaMalloc(&d_temp, bytes);

    // Copy input matrix A into d_base
    // (A is already on device, so use device->device copy)
    cudaMemcpy(d_base, A, bytes, cudaMemcpyDeviceToDevice);

    // Configure a 16×16 launch
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    if (P == 0) {
        // identity: build I on host and push to B
        __nv_bfloat16* h_I = (__nv_bfloat16*)malloc(bytes);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                h_I[i * N + j] = __float2bfloat16((i == j) ? 1.0f : 0.0f);
            }
        }
        cudaMemcpy(B, h_I, bytes, cudaMemcpyHostToDevice);
        free(h_I);
    } else if (P == 1) {
        // just A^1 = A
        cudaMemcpy(B, d_base, bytes, cudaMemcpyDeviceToDevice);
    } else {
        // First compute A^2 into d_result
        matrix_mul_bf16_scalar_kernel_optimized<<<grid, block>>>(
            d_base, d_base, d_result, N);

        // Repeatedly multiply by A for powers 3..P
        for (int i = 2; i < P; ++i) {
            matrix_mul_bf16_scalar_kernel_optimized<<<grid, block>>>(
                d_result, d_base, d_temp, N);
            // swap d_result <-> d_temp
            __nv_bfloat16* tmp = d_result;
            d_result = d_temp;
            d_temp = tmp;
        }

        // Copy final power back to user’s B
        cudaMemcpy(B, d_result, bytes, cudaMemcpyDeviceToDevice);
    }

    // Cleanup
    cudaFree(d_base);
    cudaFree(d_result);
    cudaFree(d_temp);

    // Make sure all kernels are done before returning
    cudaDeviceSynchronize();
}
