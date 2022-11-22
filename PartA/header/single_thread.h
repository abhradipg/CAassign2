#include<immintrin.h>
// Optimize this function

void singleThread(int N, int *matA, int *matB, int *output)
{
  assert( N>=4 and N == ( N &~ (N-1)));
    int sizeA=N*N;
    int sizeC=sizeA>>2;
    int noOfRowC=N>>1;
    int indexC,sum,rowFirstIndex,rowSecondIndex,rowBIndex;
    __m256i zeroVector=_mm256_set1_epi32(0);
    __m256i permuteVector=_mm256_setr_epi32(0,1,4,5,2,3,6,7);
    __m256i colFirst,colSecond,row,sumTotal;
  for(indexC=0;indexC<sizeC;indexC+=8){
    _mm256_storeu_si256((__m256i *)&output[indexC],zeroVector);
  }
  if(N<16){
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
  for(int rowA = 0,rowC = 0; rowA < N; rowA+=2, rowC+=noOfRowC) {
    rowFirstIndex=rowA * N;
    rowSecondIndex=(rowA+1) * N;
    for(int iter = 0; iter < N; iter ++){
      row = _mm256_set1_epi32(matA[rowFirstIndex + iter]+matA[rowSecondIndex + iter]);
      rowBIndex=iter * N;
      for(int colB = 0, indexC = rowC; colB < N; colB+=16, indexC+=8) 
      { 
        colFirst = _mm256_loadu_si256((__m256i *)&matB[rowBIndex + colB]);
        colSecond = _mm256_loadu_si256((__m256i *)&matB[rowBIndex + (colB+8)]);
        sumTotal = _mm256_hadd_epi32(_mm256_mullo_epi32(row,colFirst),_mm256_mullo_epi32(row,colSecond));
        _mm256_storeu_si256((__m256i *)&output[indexC],_mm256_add_epi32(_mm256_loadu_si256((__m256i *)&output[indexC]),sumTotal));
      }
    }
  }
  for(indexC=0;indexC<sizeC;indexC+=8){
    _mm256_storeu_si256((__m256i *)&output[indexC],_mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i *)&output[indexC]), permuteVector));
  }
}