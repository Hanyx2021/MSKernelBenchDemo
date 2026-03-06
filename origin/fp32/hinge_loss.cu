#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <optional>
#include <algorithm>
#include <random>
#include <cmath>
#include <vector>
#include <functional>
 
__global__ void hinge_loss_kernel(
    float* loss,
    const float* predictions,
    const int* targets,
    int N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float pred = predictions[idx];
        int target = targets[idx];
        float y = (target == 1) ? 1.0f : -1.0f;
        float sample_loss = fmaxf(0.0f, 1.0f - y * pred);
        loss[idx] = sample_loss;
    }
}
 
extern "C" void hinge_loss(
    float* loss,
    const float* predictions,
    const int* targets,
    int N) {
    
    dim3 block(1024, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, 1, 1);
    
    hinge_loss_kernel<<<grid, block>>>(loss, predictions, targets, N);
    cudaDeviceSynchronize();
}