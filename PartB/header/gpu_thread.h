#include <vector>

// Create other necessary functions here

__global__ void matrixRedMul(const int *a, const int *b, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int temp=0;
    if((threadIdx.x&1==0)&&(threadIdx.y&1==0))
       c[(row * N)>>2 + col>>1] = 0;
    for (int iter = 0; iter < N; iter++) {
       temp += a[row * N + iter] * b[iter * N + col];
    }
    c[(row * N)>>2 + col>>1]+=temp;

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
    int BLOCKS = N / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    matrixRedMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaMemcpy(matC, d_c, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
