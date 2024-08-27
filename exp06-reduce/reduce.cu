#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>    // for abs()
#include <cstdlib>  // for rand() and srand()
#include <ctime>    // for time()
using namespace std;

// __device__ __forceinline__ void WarpReduce(volatile float *data, int tid) {
//     data[tid] += data[tid + 32];
//     data[tid] += data[tid + 16];
//     data[tid] += data[tid + 8];
//     data[tid] += data[tid + 4];
//     data[tid] += data[tid + 2];
//     data[tid] += data[tid + 1];
//   }
// extern __shared__ float dynamic_share_buf[];
// __global__ void reduce_kernel(float* d,
//                             const float* a,
//                             const int n,
//                             const int m) {
//     float *data = dynamic_share_buf;
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     data[tid] = 0.0f;
//     for (int id = tid; id < m; id+=blockDim.x){
//         data[tid] += a[bid*m+id];
//     }
//     // data[tid] = tid < m ?(float) a[bid*m+tid] : 0.0f;
//     __syncthreads();
//     for(int s = blockDim.x/2; s>32; s>>=1){
//         if (tid < s) data[tid] += data[tid+s];
//         __syncthreads();
//     }
//     if (tid<32) WarpReduce(data,tid);
//     if (tid == 0) d[bid] = data[0];
// }

__device__ __forceinline__ float WarpReduceUseShuffle(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16, 32);
    v += __shfl_xor_sync(0xffffffff, v, 8, 32);
    v += __shfl_xor_sync(0xffffffff, v, 4, 32);
    v += __shfl_xor_sync(0xffffffff, v, 2, 32);
    v += __shfl_xor_sync(0xffffffff, v, 1, 32);
    return v;
  }

__global__ void reduce_kernel(float* d,
                            const float* a,
                            const int n,
                            const int m) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float v = 0.0f;
    for (int id = tid; id < m; id+=blockDim.x){
        v += a[bid*m+id];
    }
    __syncthreads();
    auto sum = WarpReduceUseShuffle(v);
    __shared__ float buf[32];
    int wid = threadIdx.x >> 5;
    int lane_id = tid & 0x1f; // = tid % 32
    if (lane_id == 0) buf[wid] = sum;
    __syncthreads();
    if (tid < 32) {
        v = tid < (blockDim.x >> 5) ? buf[tid] : 0.0f;
        sum = WarpReduceUseShuffle(v);
        if (tid == 0) d[bid] = sum;
    }
}

void launch_reduce(float* d,
                 const float* a,
                 int n,
                 int m) {
    int block = 1024;
    int share_mem_size = sizeof(float) * block;
    dim3 grid(n);
    reduce_kernel<<<grid, block, share_mem_size>>>(d, a, n, m);
}

int main() {
    const int n = 128;
    const int m = 1024;
    bool VERBOSE = false;
    float arr[n][m];
    float gather[n];

    srand(static_cast<unsigned int>(2333));
    // srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            arr[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *remote_a, *remote_t;
    cudaMalloc(&remote_a, n * m * sizeof(float));
    cudaMalloc(&remote_t, n * sizeof(float));

    cudaMemcpy(remote_a, arr, n * m * sizeof(float), cudaMemcpyHostToDevice);

    auto start = chrono::high_resolution_clock::now();
    launch_reduce(remote_t, remote_a, n, m);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = (end - start)*1000;
    cout << "running time: " << duration.count() << " 毫秒" << endl;

    cudaMemcpy(gather, remote_t, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        float tmp = 0;
        for (int j = 0; j < m; ++j) {
            tmp += arr[i][j];
        }
        if((abs(tmp - gather[i]) > 5e-2)) cout<<"error, real is "<< tmp <<", but get " << gather[i] <<endl;
    }

    if (VERBOSE){
        cout << "Raw: " << endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cout << arr[i][j] << " ";
            }
            cout << endl;
        }

        cout << "Result: " << endl;
        for (int j = 0; j < n; ++j) {
            cout << gather[j] << " ";
            cout << endl;
        }

    }

    cudaFree(remote_a);
    cudaFree(remote_t);

    return 0;
}

// nvcc -o reduce.o  reduce.cu && ./reduce.o
// nvcc -o reduce.o  reduce.cu && ncu --set full -o ncu_profile -f ./reduce.o
