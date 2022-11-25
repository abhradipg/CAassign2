#include <vector>

// Shared memory size 32*32*4
const int SHMEM_SIZE = 32*32*2;

// Create other necessary functions here

 __global__ void matrixRedMul(int *a, int *b, int *c, int N) {
    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    int row=rowC<<1;
    int col=colC<<1;
    int temp=0;

    if( row>=N || col >=N ) return;

    for (int iter = 0; iter < N; iter++) {
       temp += a[row * N + iter] * b[iter * N + col];
       temp += a[row * N + iter] * b[iter * N + col+1];
       temp += a[(row+1) * N + iter] * b[iter * N + col];
       temp += a[(row+1) * N + iter] * b[iter * N + col+1];
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

/* ijk
__global__ void matrixRedMul(int *a, int *b, int *c, int N) {
    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    int row=rowC<<1;
    int col=colC<<1;
    int temp=0;
    for (int iter = 0; iter < N; iter++) {
       temp += a[row * N + iter] * b[iter * N + col];
       temp += a[row * N + iter] * b[iter * N + col+1];
       temp += a[(row+1) * N + iter] * b[iter * N + col];
       temp += a[(row+1) * N + iter] * b[iter * N + col+1];
    }
    }
    c[rowC*(N>>1) + colC]+=temp;

}*/

/* ijk unrolled
__global__ void matrixRedMul(int *a, int *b, int *c, int N) {
    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    int row=rowC<<1;
    int col=colC<<1;
    int temp=0;
    
    if( row>=N || col >=N ) return;

    //unroll loop 4 times
    for (int iter = 0; iter < N; iter+=4) {
       int2 b_temp1 = reinterpret_cast<int2*>(&b[iter * N + col])[0];
       int2 b_temp2 = reinterpret_cast<int2*>(&b[(iter+1) * N + col])[0];
       int2 b_temp3 = reinterpret_cast<int2*>(&b[(iter+2) * N + col])[0];
       int2 b_temp4 = reinterpret_cast<int2*>(&b[(iter+3) * N + col])[0];

       int4 a_temp1 = reinterpret_cast<int4*>(&a[row * N + iter])[0];
       int4 a_temp2 = reinterpret_cast<int4*>(&a[(row+1) *N + iter])[0];

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

__global__ void matrixRedMul(int *a, int *b, int *c, int N) {
    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    int row=rowC<<1;
    int col=colC<<1;
    
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];
    
    if( row>=N || col >=N ) return;

    int temp=0;
    int blockx=blockDim.x*2;
    for (int i = 0; i < N>>1; i += blockDim.x) {
        
        int2 b_temp1 = reinterpret_cast<int2*>(&b[(i*2) * N + (threadIdx.y * 2) * N + col])[0];
        int2 b_temp2 = reinterpret_cast<int2*>(&b[(i*2 + 1) * N + (threadIdx.y * 2) * N + col])[0];
        int2 a_temp1 = reinterpret_cast<int2*>(&a[row * N + (i*2) + (threadIdx.x*2)])[0];
        int2 a_temp2 = reinterpret_cast<int2*>(&a[(row+1) * N + (i*2) + (threadIdx.x*2)])[0];

        s_a[(threadIdx.y) * blockx + (threadIdx.x*2)] = a_temp1.x+a_temp2.x;
        s_a[(threadIdx.y) * blockx + (threadIdx.x*2) + 1] = a_temp1.y+a_temp2.y;
        s_b[threadIdx.y * 2 * blockDim.x + threadIdx.x] = b_temp1.x+b_temp1.y;
        s_b[(threadIdx.y * 2 + 1) * blockDim.x + threadIdx.x] = b_temp2.x+b_temp2.y;

    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int iter = 0; iter < blockx; iter++) {
      temp  +=s_a[(threadIdx.y) * blockx + iter] * s_b[iter * blockDim.x + threadIdx.x];
    }
    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }
    c[rowC*(N>>1) + colC]+=temp;

}

*/
