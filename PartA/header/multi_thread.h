#include <pthread.h>

//Class to pass parameters to thread
class Arguments{
   int N,blockSize,blockStart;
   int *matA, *matB, *output;
   bool isLastBlock;

   public:
   Arguments(int N, int *matA,int *matB, int *output,int blockSize, int blockStart,bool isLastBlock){
      this->N=N;
      this->matA=matA;
      this->matB=matB;
      this->output=output;
      this->blockSize=blockSize;
      this->blockStart=blockStart;
      this->isLastBlock=isLastBlock;
   }
   int getN(){
    return this->N;
   }
   int getBlockSize(){
    return this->blockSize;
   }
   int getBlockStart(){
    return this->blockStart;
   }
   int* getMatA(){
    return this->matA;
   }
   int* getMatB(){
    return this->matB;
   }
   int* getOutput(){
    return this->output;
   }
   bool getIsLastBlock(){
    return this->isLastBlock;
   }
};


// Create other necessary functions here
void * multiplyBlock(void *ptr){
    Arguments *arg=(Arguments *)ptr;
    
    int N=arg->getN();
    int blockSize=arg->getBlockSize();
    int blockStart=arg->getBlockStart();
    int *matA=arg->getMatA();
    int *matB=arg->getMatB();
    int *output=arg->getOutput();
    int sizeA=N*N;
    int sizeC=sizeA>>2;
    int noOfRowC=N>>1;
    int indexC,sum,rowFirstIndex,rowSecondIndex,rowBIndex;
    bool isLastBlock=arg->getIsLastBlock();
    __m256i zeroVector=_mm256_set1_epi32(0);
    __m256i permuteVector=_mm256_setr_epi32(0,1,4,5,2,3,6,7);
    __m256i colFirst,colSecond,row,sumTotal;
    int blockEnd=isLastBlock?N:blockStart+blockSize;
    int rowCStart=(blockStart>>1)*noOfRowC,rowCEnd=(blockEnd>>1)*noOfRowC;
    for(indexC=rowCStart;indexC<rowCEnd;indexC+=8){
    _mm256_storeu_si256((__m256i *)&output[indexC],zeroVector);
  }
    for(int rowA = blockStart, rowC = rowCStart; rowA < blockEnd; rowA+=2,rowC+=noOfRowC){
      rowFirstIndex=rowA * N;
      rowSecondIndex=(rowA+1) * N;
      for(int iter = 0; iter < N; iter++){
        row = _mm256_set1_epi32(matA[rowFirstIndex + iter]+matA[rowSecondIndex + iter]);
        rowBIndex=iter * N;
        for(int colB = 0, indexC = rowC; colB < N; colB+=16, indexC+=8){
          colFirst = _mm256_loadu_si256((__m256i *)&matB[rowBIndex + colB]);
          colSecond = _mm256_loadu_si256((__m256i *)&matB[rowBIndex + (colB+8)]);
          sumTotal = _mm256_hadd_epi32(_mm256_mullo_epi32(row,colFirst),_mm256_mullo_epi32(row,colSecond));
        _mm256_storeu_si256((__m256i *)&output[indexC],_mm256_add_epi32(_mm256_loadu_si256((__m256i *)&output[indexC]),sumTotal));
         }
        }
      }

    for(indexC=rowCStart;indexC<rowCEnd;indexC+=8){
    _mm256_storeu_si256((__m256i *)&output[indexC],_mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i *)&output[indexC]), permuteVector));
  }
    return NULL;
}


// Fill in this function
void multiThread(int N, int *matA, int *matB, int *output)
{
    assert( N>=4 and N == ( N &~ (N-1)));
    const int noOfThreads=12;
    int blockSize=N/noOfThreads,blockStart=0;
    if(blockSize&1)    //Make blockSize even
      {
        blockSize--;
      }
    bool isLastBlock=true;
    if(N<16||blockSize<2){
      for(int rowA = 0; rowA < N; rowA +=2) {
      for(int colB = 0; colB < N; colB += 2){
        int sum = 0;
        for(int iter = 0; iter < N; iter++) 
        {
          sum += matA[rowA * N + iter] * matB[iter * N + colB];
          sum += matA[(rowA+1) * N + iter] * matB[iter * N + colB];
          sum += matA[rowA * N + iter] * matB[iter * N + (colB+1)];
          sum += matA[(rowA+1) * N + iter] * matB[iter * N + (colB+1)];
        }

      // compute output indices
      int rowC = rowA>>1;
      int colC = colB>>1;
      int indexC = rowC * (N>>1) + colC;
      output[indexC] = sum;
    }
   }
   return;
  }
    pthread_t threadIds[noOfThreads];
    Arguments *threadVariable=(Arguments *)malloc(noOfThreads*sizeof(Arguments));
    //Starting Threads
    for(int i=0;i<noOfThreads;i++){
       isLastBlock=(i==noOfThreads-1)?true:false;
       new (&threadVariable[i]) Arguments(N, matA, matB, output, blockSize, blockStart, isLastBlock);
       pthread_create( &threadIds[i], NULL, multiplyBlock, (void *)&threadVariable[i] );
       blockStart=blockStart+blockSize;
    }
    //Waiting for Threads to complete
    for(int i=0;i<noOfThreads;i++){
      pthread_join(threadIds[i],NULL);
    }
    free(threadVariable);
}
