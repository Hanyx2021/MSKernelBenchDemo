#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized hinge loss kernel: vectorized (4 elements per thread), branchless, coalesced loads/stores
__global__ void hinge_loss_vectorized_kernel_optimized(
    float* __restrict__ loss,
    const float* __restrict__ predictions,
    const int* __restrict__ targets,
    int N) {
    // thread and grid info
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    int stride4 = gridSize * 4;

    // compute how many full vectors (of 4) we can process
    int rem = (N / 4) * 4;

    // Vectorized loop: process 4 elements per thread
    for (int i0 = tid * 4; i0 < rem; i0 += stride4) {
        // aligned vector loads via read-only cache
        float4 pred4 = __ldg(reinterpret_cast<const float4*>(predictions + i0));
        int4 targ4 = __ldg(reinterpret_cast<const int4*>(targets + i0));

        // branchless hinge computation using fmaf
        float4 out;
        float y0 = 2.0f * (float)targ4.x - 1.0f;
        out.x = fmaxf(0.0f, fmaf(-y0, pred4.x, 1.0f));
        float y1 = 2.0f * (float)targ4.y - 1.0f;
        out.y = fmaxf(0.0f, fmaf(-y1, pred4.y, 1.0f));
        float y2 = 2.0f * (float)targ4.z - 1.0f;
        out.z = fmaxf(0.0f, fmaf(-y2, pred4.z, 1.0f));
        float y3 = 2.0f * (float)targ4.w - 1.0f;
        out.w = fmaxf(0.0f, fmaf(-y3, pred4.w, 1.0f));

        // store the 4 computed losses in one coalesced write
        *reinterpret_cast<float4*>(loss + i0) = out;
    }

    // Tail loop: handle the remaining elements (1-3)
    for (int i = rem + tid; i < N; i += gridSize) {
        float pred = __ldg(predictions + i);
        int targ = __ldg(targets + i);
        float y = 2.0f * (float)targ - 1.0f;
        float sl = fmaxf(0.0f, fmaf(-y, pred, 1.0f));
        loss[i] = sl;
    }
}

extern "C" void hinge_loss_optimized(
    float* loss,
    const float* predictions,
    const int* targets,
    int N) {
    const int blockSize = 256;
    // each thread handles 4 elements, so grid must cover N/4 threads
    int gridSize = (N + blockSize * 4 - 1) / (blockSize * 4);
    dim3 block(blockSize, 1, 1);
    dim3 grid(gridSize, 1, 1);
    hinge_loss_vectorized_kernel_optimized<<<grid, block>>>(
        loss, predictions, targets, N);
    cudaDeviceSynchronize();
}