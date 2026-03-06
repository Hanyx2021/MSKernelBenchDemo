#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel for reversing an array using pure streaming loads and stores
__global__ void reverse_array_kernel_optimized(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Compute reversed index and perform one load and one store
        int revIdx = N - 1 - idx;
        float v = input[idx];
        output[revIdx] = v;
    }
}

extern "C" void reverse_array_optimized(float* output, float* input, int N) {
    // One thread per element
    const int B = 256;  // threads per block
    int blocks = (N + B - 1) / B;
    
    // Launch kernel
    reverse_array_kernel_optimized<<<blocks, B>>>(input, output, N);
    cudaDeviceSynchronize();
}