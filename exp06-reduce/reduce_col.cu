#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>    // for abs()
#include <cstdlib>  // for rand() and srand()
#include <ctime>    // for time()
using namespace std;

__device__ __forceinline__ float WarpReduceUseShuffle(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16, 32);
    v += __shfl_xor_sync(0xffffffff, v, 8, 32);
    v += __shfl_xor_sync(0xffffffff, v, 4, 32);
    v += __shfl_xor_sync(0xffffffff, v, 2, 32);
    v += __shfl_xor_sync(0xffffffff, v, 1, 32);
    return v;
  }



__global__ void reduce_kernel(float *d,const float *a, const int n,const int m){
    int n_offset = blockIdx.x * 32,
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float buf[32][32 + 1];
    float sum = 0.0f;
    int ni =n_offset + tx;
    if(ni < m){
        for(int mi = ty; mi <n; mi += 32){
            sum += a[mi * m + ni];
//             printf("tx=%d ty=%d v=%f\n", tx,ty,a[mi * m + ni] );
        }
    }
    buf[ty][tx]= sum;
    __syncthreads();
    auto v= buf[tx][ty];
    sum = WarpReduceUseShuffle(v);
    ni =n_offset + ty;
    if(ni <n && tx == 0){
        d[ni]= sum;
    }
}


void launch_reduce(float* d,
                 const float* a,
                 int n,
                 int m) {
    dim3 block (32,32);
    dim3 grid((n+31)/32,(m+31)/32);
    reduce_kernel<<<grid, block>>>(d, a, n, m);
}

int main() {
    const int n = 400;
    const int m = 20;
    bool VERBOSE = false;
    float arr[n][m];
    float gather[m];

    srand(static_cast<unsigned int>(2333));
    // srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            arr[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *remote_a, *remote_t;
    cudaMalloc(&remote_a, n * m * sizeof(float));
    cudaMalloc(&remote_t, m * sizeof(float));

    cudaMemcpy(remote_a, arr, n * m * sizeof(float), cudaMemcpyHostToDevice);

    auto start = chrono::high_resolution_clock::now();
    launch_reduce(remote_t, remote_a, n, m);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = (end - start)*1000;
    cout << "running time: " << duration.count() << " ms" << endl;

    cudaMemcpy(gather, remote_t, m * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; ++i) {
        float tmp = 0;
        for (int j = 0; j < n; ++j) {
            tmp += arr[j][i];
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
        for (int j = 0; j < m; ++j) {
            cout << gather[j] << " ";
            cout << endl;
        }

    }

    cudaFree(remote_a);
    cudaFree(remote_t);

    return 0;
}


// nvcc  reduce_col.cu -o reduce_col  -allow-unsupported-compiler
// ncu --set full -o ncu_profile -f ./reduce_col.exe