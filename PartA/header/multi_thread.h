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

void * computeB(void *ptr){
Arguments *arg=(Arguments *)ptr;
    
    int N=arg->getN();
    int blockSize=arg->getBlockSize();
    int blockStart=arg->getBlockStart();
    int *matB=arg->getMatB();
    int *output=arg->getOutput();
    int sizeA=N*N;
    int indexC,sum,rowFirstIndex,rowSecondIndex,rowBIndex,noOfRowC=N>>1;
    bool isLastBlock=arg->getIsLastBlock();
    __m256i zeroVector=_mm256_set1_epi32(0);
    __m256i permuteVector=_mm256_setr_epi32(0,1,4,5,2,3,6,7);
    __m256i colFirst,colSecond,row,sumTotal;
    int blockEnd=isLastBlock?N:blockStart+blockSize;
    for(int rowB = blockStart; rowB < blockEnd; rowB++){
        for(int col=0;col<N;col+=16){
      _mm256_storeu_si256((__m256i *)&output[rowB*(noOfRowC)+(col>>1)],_mm256_hadd_epi32(_mm256_loadu_si256((__m256i *)&matB[rowB*N+col]),
                                                                      _mm256_loadu_si256((__m256i *)&matB[rowB*N+col+8])));
    }
      }
    return NULL;
}

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
    int indexC,sum,rowFirstIndex,rowSecondIndex,rowBIndex,rowB;
    bool isLastBlock=arg->getIsLastBlock();
    __m256i zeroVector=_mm256_set1_epi32(0);
    __m256i permuteVector=_mm256_setr_epi32(0,1,4,5,2,3,6,7);
    __m256i colFirst1,colFirst2,colFirst3,colFirst4,colFirst5,colFirst6,colFirst7,colFirst8;
    __m256i sumTotal1,sumTotal2,sumTotal3,sumTotal4,sumTotal5,sumTotal6,sumTotal7,sumTotal8;
    __m256i colSecond,row,row1,row2,row3,row4,row5,row6,row7,row8,colC;
    int blockEnd=isLastBlock?N:blockStart+blockSize;
    int rowCStart=(blockStart>>1)*noOfRowC,rowCEnd=(blockEnd>>1)*noOfRowC;
    for(indexC=rowCStart;indexC<rowCEnd;indexC+=8){
    _mm256_storeu_si256((__m256i *)&output[indexC],zeroVector);
  }
    for(int rowA = blockStart, rowC = rowCStart; rowA < blockEnd; rowA+=2,rowC+=noOfRowC){
      rowFirstIndex=rowA * N;
      rowSecondIndex=(rowA+1) * N;
      for(int iter = 0; iter < N; iter +=8){
      row = _mm256_add_epi32(_mm256_loadu_si256((__m256i *)&matA[rowFirstIndex + iter]),
                             _mm256_loadu_si256((__m256i *)&matA[rowSecondIndex + iter]));
      int *r=(int *)&row;
      row1 = _mm256_set1_epi32(r[0]);
      row2 = _mm256_set1_epi32(r[1]);
      row3 = _mm256_set1_epi32(r[2]);
      row4 = _mm256_set1_epi32(r[3]);
      row5 = _mm256_set1_epi32(r[4]);
      row6 = _mm256_set1_epi32(r[5]);
      row7 = _mm256_set1_epi32(r[6]);
      row8 = _mm256_set1_epi32(r[7]);
      rowBIndex=iter * noOfRowC;
      for(int colB = 0, indexC = rowC; colB < noOfRowC; colB+=8, indexC+=8) 
      { 
        colC=_mm256_loadu_si256((__m256i *)&output[indexC]);
        rowB=rowBIndex;
        colFirst1 = _mm256_loadu_si256((__m256i *)&matB[rowB + colB]);
        sumTotal1 = _mm256_mullo_epi32(row1,colFirst1);
        colC=_mm256_add_epi32(colC,sumTotal1);
        rowB+=noOfRowC;
        colFirst2 = _mm256_loadu_si256((__m256i *)&matB[rowB + colB]);
        sumTotal2 = _mm256_mullo_epi32(row2,colFirst2);
        colC=_mm256_add_epi32(colC,sumTotal2);
        rowB+=noOfRowC;
        colFirst3 = _mm256_loadu_si256((__m256i *)&matB[rowB + colB]);
        sumTotal3 = _mm256_mullo_epi32(row3,colFirst3);
        colC=_mm256_add_epi32(colC,sumTotal3);
        rowB+=noOfRowC;
        colFirst4 = _mm256_loadu_si256((__m256i *)&matB[rowB + colB]);
        sumTotal4 = _mm256_mullo_epi32(row4,colFirst4);
        colC=_mm256_add_epi32(colC,sumTotal4);
        rowB+=noOfRowC;
        colFirst5 = _mm256_loadu_si256((__m256i *)&matB[rowB + colB]);
        sumTotal5 = _mm256_mullo_epi32(row5,colFirst5);
        colC=_mm256_add_epi32(colC,sumTotal5);
        rowB+=noOfRowC;
        colFirst6 = _mm256_loadu_si256((__m256i *)&matB[rowB + colB]);
        sumTotal6 = _mm256_mullo_epi32(row6,colFirst6);
        colC=_mm256_add_epi32(colC,sumTotal6);
        rowB+=noOfRowC;
        colFirst7 = _mm256_loadu_si256((__m256i *)&matB[rowB + colB]);
        sumTotal7 = _mm256_mullo_epi32(row7,colFirst7);
        colC=_mm256_add_epi32(colC,sumTotal7);
        rowB+=noOfRowC;
        colFirst8 = _mm256_loadu_si256((__m256i *)&matB[rowB + colB]);
        sumTotal8 = _mm256_mullo_epi32(row8,colFirst8);
        _mm256_storeu_si256((__m256i *)&output[indexC],_mm256_add_epi32(colC,sumTotal8));
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
    const int noOfThreads=8;
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
    pthread_t threadId1s[noOfThreads];
    pthread_t threadId2s[noOfThreads];
    Arguments *threadVariablePreCompute=(Arguments *)malloc(noOfThreads*sizeof(Arguments));
    //Starting Threads
    Arguments *threadVariable=(Arguments *)malloc(noOfThreads*sizeof(Arguments));
    int *matmatB = new int[N*N>>1];
    for(int i=0;i<noOfThreads;i++){
       isLastBlock=(i==noOfThreads-1)?true:false;
       new (&threadVariablePreCompute[i]) Arguments(N, matA, matB, matmatB, blockSize, blockStart, isLastBlock);
       pthread_create( &threadId1s[i], NULL, computeB, (void *)&threadVariablePreCompute[i] );
       blockStart=blockStart+blockSize;
    }
    //Waiting for Threads to complete
    for(int i=0;i<noOfThreads;i++){
      pthread_join(threadId1s[i],NULL);
    }
    free(threadVariablePreCompute);
    blockStart=0;
    for(int i=0;i<noOfThreads;i++){
       isLastBlock=(i==noOfThreads-1)?true:false;
       new (&threadVariable[i]) Arguments(N, matA, matmatB, output, blockSize, blockStart, isLastBlock);
       pthread_create( &threadId2s[i], NULL, multiplyBlock, (void *)&threadVariable[i] );
       blockStart=blockStart+blockSize;
    }
    //Waiting for Threads to complete
    for(int i=0;i<noOfThreads;i++){
      pthread_join(threadId2s[i],NULL);
    }
    free(threadVariable);
    free(matmatB);
}
