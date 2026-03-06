#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Optimized configuration constants
static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 64;
static constexpr int BLOCK_K = 32;
static constexpr int THREADS_X = 16;
static constexpr int THREADS_Y = 16;
static constexpr int REG_M = BLOCK_M / THREADS_Y;  // 4
static constexpr int REG_N = BLOCK_N / THREADS_X;  // 4

// Optimized kernel with shared-memory tiling, register blocking, and bf16->fp32 computation
__global__ void high_order_contraction_bf16_kernel_optimized(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim)
{
    // Compute flattened dimensions
    int M = a_dim * b_dim;
    int K = x_dim * y_dim;
    int N = c_dim;

    // Block indices
    int block_m = blockIdx.x;
    int block_n = blockIdx.y;
    int m_start = block_m * BLOCK_M;
    int n_start = block_n * BLOCK_N;

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Allocate shared memory for A and B tiles in bf16
    extern __shared__ __nv_bfloat16 shared_mem[];
    __nv_bfloat16* As = shared_mem;                             // size BLOCK_M * BLOCK_K
    __nv_bfloat16* Bs = shared_mem + BLOCK_M * BLOCK_K;        // size BLOCK_K * BLOCK_N

    // Registers for accumulation
    float acc[REG_M][REG_N];
    for(int i = 0; i < REG_M; ++i)
        for(int j = 0; j < REG_N; ++j)
            acc[i][j] = 0.0f;

    // Each thread cooperatively load tiles and compute
    for(int k_start = 0; k_start < K; k_start += BLOCK_K) {
        // Load A tile into shared memory
        int n_elements_A = BLOCK_M * BLOCK_K;
        int tid = ty * THREADS_X + tx;
        int n_threads = THREADS_X * THREADS_Y;
        for(int idx = tid; idx < n_elements_A; idx += n_threads) {
            int m_local = idx / BLOCK_K;
            int k_local = idx % BLOCK_K;
            int m_global = m_start + m_local;
            int k_global = k_start + k_local;
            if(m_global < M && k_global < K) {
                // map m_global -> (a,b)
                int a = m_global / b_dim;
                int b = m_global % b_dim;
                // map k_global -> (x,y)
                int x = k_global / y_dim;
                int y = k_global % y_dim;
                int idxA = ((a * x_dim + x) * b_dim + b) * y_dim + y;
                As[m_local * BLOCK_K + k_local] = A[idxA];
            } else {
                As[m_local * BLOCK_K + k_local] = __float2bfloat16(0.0f);
            }
        }
        // Load B tile into shared memory
        int n_elements_B = BLOCK_K * BLOCK_N;
        for(int idx = tid; idx < n_elements_B; idx += n_threads) {
            int k_local = idx / BLOCK_N;
            int n_local = idx % BLOCK_N;
            int k_global = k_start + k_local;
            int n_global = n_start + n_local;
            if(k_global < K && n_global < N) {
                // map k_global -> (x,y)
                int x = k_global / y_dim;
                int y = k_global % y_dim;
                int c = n_global;
                int idxB = ((x * c_dim + c) * y_dim + y);
                Bs[k_local * BLOCK_N + n_local] = B[idxB];
            } else {
                Bs[k_local * BLOCK_N + n_local] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Register-level blocking
        int row_start = ty * REG_M;
        int col_start = tx * REG_N;
        for(int kk = 0; kk < BLOCK_K; ++kk) {
            // Load A regs
            float a_reg[REG_M];
            for(int i = 0; i < REG_M; ++i) {
                __nv_bfloat16 av = As[(row_start + i) * BLOCK_K + kk];
                a_reg[i] = __bfloat162float(av);
            }
            // Load B regs and compute
            for(int j = 0; j < REG_N; ++j) {
                __nv_bfloat16 bv = Bs[kk * BLOCK_N + (col_start + j)];
                float b_reg = __bfloat162float(bv);
                for(int i = 0; i < REG_M; ++i) {
                    acc[i][j] += a_reg[i] * b_reg;
                }
            }
        }
        __syncthreads();
    }

    // Write back C
    int row_start = ty * REG_M;
    int col_start = tx * REG_N;
    for(int i = 0; i < REG_M; ++i) {
        int m_local = row_start + i;
        int m_global = m_start + m_local;
        if(m_global >= M) continue;
        int a = m_global / b_dim;
        int b = m_global % b_dim;
        for(int j = 0; j < REG_N; ++j) {
            int n_local = col_start + j;
            int n_global = n_start + n_local;
            if(n_global >= N) continue;
            int c = n_global;
            int idxC = ((a * b_dim + b) * c_dim + c);
            C[idxC] = __float2bfloat16(acc[i][j]);
        }
    }
}

// External C wrapper invoking the optimized kernel
extern "C" void high_order_contraction_bf16_optimized(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int a_dim, int b_dim, int c_dim,
    int x_dim, int y_dim)
{
    // Flattened sizes
    int M = a_dim * b_dim;
    int N = c_dim;
    dim3 threads(THREADS_X, THREADS_Y);
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M,
              (N + BLOCK_N - 1) / BLOCK_N);
    // Shared memory size
    size_t shared_bytes = (size_t)(BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(__nv_bfloat16);

    high_order_contraction_bf16_kernel_optimized<<<grid, threads, shared_bytes>>>(
        A, B, C,
        a_dim, b_dim, c_dim,
        x_dim, y_dim
    );
    cudaDeviceSynchronize();
}
