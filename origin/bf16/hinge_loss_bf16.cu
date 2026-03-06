#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <optional>
#include <algorithm>
#include <random>
#include <cmath>
#include <vector>
#include <functional>
 
__global__ void hinge_loss_bf16_kernel(
    __nv_bfloat16* loss,
    const __nv_bfloat16* predictions,
    const int* targets,
    int N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float pred = __bfloat162float(predictions[idx]);
        int target = targets[idx];
        float y = (target == 1) ? 1.0f : -1.0f;
        float sample_loss = fmaxf(0.0f, 1.0f - y * pred);
        loss[idx] = __float2bfloat16(sample_loss);
    }
}
 
extern "C" void hinge_loss_bf16(
    __nv_bfloat16* loss,
    const __nv_bfloat16* predictions,
    const int* targets,
    int N) {
    
    dim3 block(1024, 1, 1);
    dim3 grid(1, 1, 1);
    
    hinge_loss_bf16_kernel<<<grid, block>>>(loss, predictions, targets, N);
    cudaDeviceSynchronize();
}