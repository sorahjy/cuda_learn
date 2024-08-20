
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cstdlib>  // for rand() and srand()
#include <ctime>    // for time()
using namespace std;

// __global__ void transpose_kernel(float* d,
//     const float* a,
//     const int n,
//     const int m) {
//     int row_base = blockIdx.y * blockDim.y;
//     int col_base = blockIdx.x * blockDim.x;
//     int ty = threadIdx.y;
//     int tx = threadIdx.x;
//     int row_id = row_base + ty;
//     int col_id = col_base + tx;
//     if (row_id < n && col_id < m ){
//         d[col_id * n + row_id] = a[row_id * m + col_id];
//     }
// }

__global__ void transpose_kernel(float* d,
                            const float* a,
                            const int n,
                            const int m) {
    // __shared__ float data[32][32];  // bank conflict
    __shared__ float data[32][33];
    int row_base = blockIdx.y * blockDim.y;
    int col_base = blockIdx.x * blockDim.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row_id = row_base + ty;
    int col_id = col_base + tx;
    data[ty][tx] = (row_id < n && col_id < m) ? a[row_id * m + col_id] : 0.0f;
    __syncthreads();
    row_id = row_base + tx;
    col_id = col_base + ty;
    if (row_id < n && col_id < m ){
        d[col_id * n + row_id] = data[tx][ty];
    }
}


void launch_transpose(float* d,
                 const float* a,
                 int n,
                 int m) {
    dim3 block(32, 32);
    dim3 grid((m + 31)>>5, (n + 31)>>5);
    transpose_kernel<<<grid, block>>>(d, a, n, m);
}


int main() {
    const int n = 3;
    const int m = 40;
    bool VERBOSE = false;
    float arr[n][m];
    float tran[m][n];

    srand(static_cast<unsigned int>(2333));
    // srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            arr[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *remote_a, *remote_t;
    cudaMalloc(&remote_a, n * m * sizeof(float));
    cudaMalloc(&remote_t, m * n * sizeof(float));

    cudaMemcpy(remote_a, arr, n * m * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    launch_transpose(remote_t, remote_a, n, m);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (end - start)*1000;
    std::cout << "running time: " << duration.count() << " ms" << std::endl;

    cudaMemcpy(tran, remote_t, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if(arr[i][j]!=tran[j][i]) cout<<"error"<<endl;
        }
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
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                cout << tran[i][j] << " ";
            }
            cout << endl;
        }
    }

    cudaFree(remote_a);
    cudaFree(remote_t);

    return 0;
}

// nvcc -o transpose.o  transpose.cu && ncu --set full -o ncu_profile -f ./transpose.o
// nvcc -o transpose.o  transpose.cu && ./transpose.o
// nvcc  transpose.cu -o transpose  -allow-unsupported-compiler
// ncu --set full -o ncu_profile -f ./transpose.exe