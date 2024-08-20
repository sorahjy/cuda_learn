
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void add2_kernel(float* d,
                            const float* a,
                            const float* b,
                            const float* c,
                            const int n,
                            const int m) {
//     n =3 , m =4
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * m){
        d[i] = a[i/m] * b[0] + c[i%m];
//         printf("d[%d]%f:=a[%d]:%f * b[%d]:%f + c[%d]:%f\n",i,d[i],i%m,a[i%m],i,b[i],i/n,c[i/n]);
    }

}

void launch_add2(float* d,
                 const float* a,
                 const float* b,
                 const float* c,
                 int n,
                 int m) {
    dim3 block_size(128);
    dim3 num_blocks((n*m + block_size.x - 1) / block_size.x);
    add2_kernel<<<num_blocks, block_size>>>(d, a, b,c, n,m);
}


int main() {
    const int n = 3;
    const int m = 4;

    float h_a[n] = {1.0, 2.0, 3.0};
    float h_b[1] = {2.0};
    float h_c[m] = {0.1, 0.2, 0.3, 0.4};
    float h_d[n][m];

    float *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, 1 * sizeof(float));
    cudaMalloc(&d_c, m * sizeof(float));
    cudaMalloc(&d_d, n * m * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, m * sizeof(float), cudaMemcpyHostToDevice);

    launch_add2(d_d, d_a, d_b, d_c, n, m);

    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << h_d[i][j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    return 0;
}


// whidows compile:  nvcc  tmp.cu -o tmp_exe  -allow-unsupported-compiler
// linux compile: nvcc -o tmp.o tmp.cu
// nsys profile ./tmp.o
// ncu --set full -o ncu_profile ./tmp.o