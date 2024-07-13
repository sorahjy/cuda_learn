// #include <stdio.h>
//
// __global__ void add2_kernel(float* d,
//                             const float* a,
//                             const float* b,
//                             const float* c,
//                             const int n,
//                             const int m) {
//
// //     for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
// //             i < n; i += gridDim.x * blockDim.x) {
// //         c[i] = a[i] + b[i];
// //     }
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;
//     auto j = blockIdx.y * blockDim.y + threadIdx.y;
//     if (i < n && j < m){
//         auto index = n * j + i;
//         c[index] = a[index] + b[index];
// //         printf("%d:a=%f,b=%f,c=%f\n",index,a[index],b[index],c[index]);
//     }
//
// }
//
//
// void launch_add2(float* d,
//                  const float* a,
//                  const float* b,
//                  const float* c,
//                  int n,
//                  int m) {
//     dim3 block_size(16,16);
//     dim3 num_blocks((n + block_size.x - 1) / block_size.x,(m + block_size.y - 1) / block_size.y);
//     add2_kernel<<<num_blocks, block_size>>>(c, a, b, n,m);
// }

// #include <stdio.h>

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

