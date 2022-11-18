#include <vector>

// Create other necessary functions here

__global__ void matrixRedMul(int *a, int *b, int *c, int N) {
    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    int row=rowC<<1;
    int col=colC<<1;
    int temp=0;
    //unroll loop 4 times
    for (int iter = 0; iter < N; iter+=4) {
       int2 b_temp1 = reinterpret_cast<int2*>(&b[iter * N + col])[0];
       int2 b_temp2 = reinterpret_cast<int2*>(&b[(iter+1) * N + col])[0];
       int2 b_temp3 = reinterpret_cast<int2*>(&b[(iter+2) * N + col])[0];
       int2 b_temp4 = reinterpret_cast<int2*>(&b[(iter+3) * N + col])[0];

       int4 a_temp1 = reinterpret_cast<int4*>(&a[row * N + iter])[0];
       int4 a_temp2 = reinterpret_cast<int4*>(&a[(row+1) + iter])[0];

       temp += a_temp1.x * b_temp1.x;
       temp += a_temp1.x * b_temp1.y;
       temp += a_temp2.x * b_temp1.x;
       temp += a_temp2.x * b_temp1.y;

       temp += a_temp1.y * b_temp2.x;
       temp += a_temp1.y * b_temp2.y;
       temp += a_temp2.y * b_temp2.x;
       temp += a_temp2.y * b_temp2.y;

       temp += a_temp1.z * b_temp3.x;
       temp += a_temp1.z * b_temp3.y;
       temp += a_temp2.z * b_temp3.x;
       temp += a_temp2.z * b_temp3.y;

       temp += a_temp1.w * b_temp4.x;
       temp += a_temp1.w * b_temp4.y;
       temp += a_temp2.w * b_temp4.x;
       temp += a_temp2.w * b_temp4.y;

    }
    c[rowC*(N>>1) + colC]+=temp;

}

// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *matC)
{
    int sizeA = N*N*sizeof(int);
    int sizeC = sizeA>>2;
    
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeA);
    cudaMalloc(&d_b, sizeA);
    cudaMalloc(&d_c, sizeC);
  
    cudaMemcpy(d_a, matA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, matB, sizeA, cudaMemcpyHostToDevice);

    int THREADS = 32;
    int BLOCKS = (N>>1) / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    matrixRedMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaMemcpy(matC, d_c, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

/* ijk improv
__global__ void matrixRedMul(int *a, int *b, int *c, int N) {
    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    int row=rowC<<1;
    int col=colC<<1;
    int temp=0;
    for (int iter = 0; iter < N; iter++) {
       int2 b_temp = reinterpret_cast<int2*>(&b[iter * N + col])[0];
       temp += a[row * N + iter] * b_temp.x;
       temp += a[row * N + iter] * b_temp.y;
       temp += a[(row+1) * N + iter] * b_temp.x;
       temp += a[(row+1) * N + iter] * b_temp.y;
    }
    c[rowC*(N>>1) + colC]+=temp;

}*/