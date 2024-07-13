#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

__global__ void add2_kernel(float** d,
                            float** a,
                            float b,
                            float* c,
                            const int n,
                            const int m) {
    //     n =3 , m =4 , a (3,1)  b(1)  c(4)
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * m){
        d[i/m][i%m] = a[i/m][0] * b + c[i%m];
        // printf("d[%d]%f:=a[%d]:%f * b[%d]:%f + c[%d]:%f\n",i,d[i][i%m],i%m,a[i/m][0],i,b,i/n,c[i%m]);

    }
}
// cudaMallocPitch cudaMemset2D
int main()
{
    auto n = 4;
    auto m = 5;
    float B = 1.0;
    float** A;
    float* C;
    float** D;
    cudaMalloc(&A, sizeof(float*) * n);
    cudaMalloc(&D, sizeof(float*) * n);
    cudaMalloc(&C, sizeof(float) * m);
    float** a_rows = (float**)malloc(sizeof(float*) * n);
    float** d_rows = (float**)malloc(sizeof(float*) * n);

    vector<float> c_cpu(m,0);
    vector<vector<float>> a_cpu(n,vector<float>(1,0));
    vector<vector<float>> d_cpu(n,vector<float>(m,0));
    for (int i = 0; i < m ; ++i) {
        c_cpu[i] = 0.1f * i;
    }
    for (int i = 0; i < n ; ++i) {
        a_cpu[i][0] = 0.5f * i;
        cudaMalloc(&a_rows[i], sizeof(float) * 1);
        cudaMalloc(&d_rows[i], sizeof(float) * m);
        // cudaMemcpy(A[i], a_cpu[i].data(), sizeof(float)*1, cudaMemcpyHostToDevice); 这里写错了
        cudaMemcpy(a_rows[i], a_cpu[i].data(), sizeof(float)*1, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(C, c_cpu.data(), sizeof(float)*m, cudaMemcpyHostToDevice);
    cudaMemcpy(A, a_rows, sizeof(float*) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(D, d_rows, sizeof(float*) * n, cudaMemcpyHostToDevice);


    dim3 block(256);
    dim3 grid((n*m+block.x-1)/block.x);
    add2_kernel<<<grid,block>>>(D,A,B,C,n,m);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; ++i) {
        cudaMemcpy(d_cpu[i].data(), d_rows[i], sizeof(float)*m, cudaMemcpyDeviceToHost);
    }
    for (int i = 0; i < n; ++i) {
        cudaFree(a_rows[i]);
        cudaFree(d_rows[i]);
    }
    cudaFree(A);
    cudaFree(D);
    cudaFree(C);

    for (int i =0;i <n;i++) {
        for (int j =0 ;j<m;j++) printf("%.3f ",d_cpu[i][j]);
        printf("\n");
    }

    return 0;
}

// nvcc  add.cu -o add_cuda  -allow-unsupported-compiler
// nvcc -o add.o add.cu
