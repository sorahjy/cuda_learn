#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cstdlib>  // for rand() and srand()
#include <ctime>    // for time()
using namespace std;

__global__ void transpose_kernel(float* d,
                            const float* a,
                            const int y,
                            const int z) {
    // __shared__ float data[32][32];  // bank conflict
    __shared__ float data[32][33];
    int row_base = blockIdx.y * blockDim.y;
    int col_base = blockIdx.x * blockDim.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row_id = row_base + ty;
    int col_id = col_base + tx;
    auto cur_a = a + blockIdx.z * y * z; // calculate pointer
    auto cur_d = d + blockIdx.z * y * z; // calculate pointer
    data[ty][tx] = (row_id < y && col_id < z) ? cur_a[row_id * z + col_id] : 0.0f;
    __syncthreads();
    row_id = row_base + tx;
    col_id = col_base + ty;
    if (row_id < y && col_id < z ){
        cur_d[col_id * y + row_id] = data[tx][ty];
    }
}


void launch_transpose(float* d,
                 const float* a,
                 int x,
                 int y,
                 int z) {
    dim3 block(32, 32);
    dim3 grid((z + 31)>>5, (y + 31)>>5, x);
    transpose_kernel<<<grid, block>>>(d, a, y, z);
}


int main() {
    const int x = 78;
    const int y = 45;
    const int z = 98;
    bool VERBOSE = false;
    float arr[x][y][z];
    float tran[x][z][y];

    srand(static_cast<unsigned int>(2333));
    // srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0 ; k<z; ++k){
                arr[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }

    float *remote_a, *remote_t;
    cudaMalloc(&remote_a, x * y * z * sizeof(float));
    cudaMalloc(&remote_t, x * z * y * sizeof(float));
    cudaMemcpy(remote_a, arr, x * y * z * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    launch_transpose(remote_t, remote_a, x, y, z);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (end - start)*1000;
    std::cout << "running time: " << duration.count() << " 毫秒" << std::endl;

    cudaMemcpy(tran, remote_t, x * y * z * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for(int k = 0; k < z; ++k)
                if(arr[i][j][k]!=tran[i][k][j]) cout<<"error"<<endl;
        }
    }

    if (VERBOSE){
        for (int k = 0; k<x; ++k){
            cout << "Raw: " << endl;
            for (int i = 0; i < y; ++i) {
                for (int j = 0; j < z; ++j) {
                    cout << arr[k][i][j] << " ";
                }
                cout << endl;
            }

            cout << "Result: " << endl;
            for (int i = 0; i < y; ++i) {
                for (int j = 0; j < z; ++j) {
                    cout << tran[k][i][j] << " ";
                }
                cout << endl;
            }
            cout<<"-------------"<<endl;
        }

    }

    cudaFree(remote_a);
    cudaFree(remote_t);

    return 0;
}

// D0, D1, D2  -> D0, D2, D1 torch.transpose(x, 1, 2)

// nvcc -o transpose_d3_s12.o  transpose_d3_s12.cu && ./transpose_d3_s12.o


