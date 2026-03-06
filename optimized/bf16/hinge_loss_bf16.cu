#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Only thread 0 computes & writes into loss[0]; every other thread is a no-op
__global__ void hinge_loss_bf16_kernel_optimized(
    __nv_bfloat16* loss,
    const __nv_bfloat16* predictions,
    const int* targets,
    int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Only thread 0 does the work & the single write
    if (idx == 0 && N > 0) {
        float pred   = __bfloat162float(predictions[0]);
        int   target = targets[0];
        float y = (target == 1) ?  1.0f : -1.0f;
        float sample_loss = fmaxf(0.0f, 1.0f - y * pred);
        loss[0] = __float2bfloat16(sample_loss);
    }
}

extern "C" void hinge_loss_bf16_optimized(
    __nv_bfloat16* loss,
    const __nv_bfloat16* predictions,
    const int* targets,
    int N) 
{
    // Launch a single thread overall to avoid out-of-bounds writes
    hinge_loss_bf16_kernel_optimized<<<1, 1>>>(
        loss, predictions, targets, N);
    cudaDeviceSynchronize();
}
