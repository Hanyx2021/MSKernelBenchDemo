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

__global__ void conv_2d_kernel(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
   
      int col = blockDim.x * blockIdx.x + threadIdx.x;
      int row = blockDim.y * blockIdx.y + threadIdx.y;

       int output_rows = input_rows - kernel_rows + 1;
       int output_cols = input_cols - kernel_cols + 1;

      if(col < output_cols && row < output_rows) {
        float sum = 0.0f;
        for(int m = 0; m < kernel_rows; m++) {
            for(int n = 0; n < kernel_cols; n++) {
                const int input_row = row + m;
                const int input_col = col + n;
                sum += input[input_row * input_cols + input_col] * kernel[m * kernel_cols + n];
            }
        }
        output[row * output_cols + col] = sum;
      }
}

extern "C" void conv_2d(const float* input, const float* kernel, float* output, 
                      int input_rows, int input_cols, 
                      int kernel_rows, int kernel_cols) {
    
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 block(16, 16);
    
    dim3 grid((output_cols + block.x - 1) / block.x,
              (output_rows + block.y - 1) / block.y);
    
    conv_2d_kernel<<<grid, block>>>(input, kernel, output, 
                                   input_rows, input_cols, 
                                   kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}