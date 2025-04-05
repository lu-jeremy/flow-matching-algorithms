#include <cuda_runtime.h>
#include <cuda/std/limits>
#include <iostream>

using namespace std;

#define CUDA_CHECK(ans) { errHandler((ans), __FILE__, __LINE__); }
inline void errHandler(cudaError_t err, const char *file, int line, bool abort=true) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << " in" << 
            file << ":" << line << endl;
        if (abort) exit(err);
    }
}

template<int BLOCK_SIZE>
__global__ void max_pool_1d(const float *input, int k, int s, int p, int d,
        float *output, size_t H, size_t H_out) {
    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;

    extern __shared__ float shm[];

    int out_start = blockDim.x * blockIdx.x;
    int out_end = min(blockDim.x * (blockIdx.x + 1) - 1, (int) H_out - 1);

    int in_start = s * out_start - p;
    int in_end = s * out_end + d * (k - 1) - p;

    in_start = max(in_start, 0);
    in_end = min(in_end, (int) H - 1);

    // # of elements in shm
    int load_size = in_end - in_start + 1;

    for (int j = tid; j <= in_end - in_start; j += blockDim.x) {
        int m = in_start + j;
        if (m >= 0 && m < H) {
            shm[m - in_start] = input[m]; 
        }
        else {
            shm[m - in_start] = cuda::std::numeric_limits<float>::lowest();
        }
    }

    __syncthreads();  // H_out is the kernel's boundary idx

    if (i < H_out) {
        float val = cuda::std::numeric_limits<float>::lowest();

        int base_shm_idx = s * i - p - in_start;
#pragma unroll
        for (int m = 0; m < k; m++) {
           int shm_idx = base_shm_idx + d * m; 
           if (shm_idx >= 0 && shm_idx < load_size) {
               val = max(val, shm[shm_idx]);
           }
        }

        output[i] = val; 
    }

}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {    
    float *d_input;
    float *d_output;

    // think in terms of -1 * (dilation * (kernel_size - 1) + 1) 
    int H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * H));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * H_out));

    CUDA_CHECK(cudaMemcpy(d_input, input, sizeof(float) * H, cudaMemcpyHostToDevice));

    int n = H_out;
    int block_size = 512;
    int grid_size = (n + block_size - 1) / block_size;

    // max index from with block size
    int max_range = stride * block_size + dilation * (kernel_size - 1); 

    max_pool_1d<512><<<grid_size, block_size, sizeof(float) * max_range>>>(
        d_input, kernel_size, stride, padding, dilation, d_output, H, H_out);

    CUDA_CHECK(cudaMemcpy(output, d_output, sizeof(float) * H_out,
                cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input));
}

int main() {
    size_t H = 10;
    float input[10] = {1.0f, 3.0f, 2.0f, 4.0f, 6.0f, 5.0f, 8.0f, 7.0f,
        9.0f, 10.0f};

    int kernel_size = 3;
    int stride = 2;
    int padding = 1;
    int dilation = 1;

    size_t H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    float output[H_out];

    solution(input, kernel_size, stride, padding, dilation, output, H);

    cout << "Output after max pooling: ";
    for (size_t i = 0; i < H_out; i++) {
        cout << output[i] << " "; 
    }
    cout << endl;

    return 0;
}


