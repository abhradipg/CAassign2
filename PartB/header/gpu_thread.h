#include <vector>

// Create other necessary functions here

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
