#include <iostream>
#include "ttm_gpu.h"
#include <vector>


int BLOCKSIZE = 512;
dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

inline cudaError_t checkCuda(cudaError_t result, int s){

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error in line : %s - %d\n", cudaGetErrorString(result), s);
    assert(result == cudaSuccess);
  }
  return result;
}

void cuda_timer_start(cudaEvent_t start){
    checkCuda(cudaEventRecord(start), __LINE__);
}
void cuda_timer_stop(cudaEvent_t start, cudaEvent_t stop, float &mili){
    checkCuda(cudaEventRecord(stop), __LINE__);
    cudaEventSynchronize(stop);
    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
    cudaDeviceSynchronize();
}

__global__ void ttm_kernel(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
    ITYPE *fbrPtr1, unsigned int nSlices, DTYPE *dY, DTYPE *dU2, ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC){

    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 31;
    unsigned int gId = (blockIdx.x * blockDim.x + tId);
    unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
    unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
    DTYPE tmp_val;

    if(slc < nSlices){ 
                                      
        unsigned int mappedSlc = dSlcMapperBin[slc];
        unsigned int idx0 = dfbrIdx0[mappedSlc] ;//slc;
        int fb_st = fbrPtr0[mappedSlc];
        int fb_end = fbrPtr0[mappedSlc+1];

        for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
            tmp_val = 0;
            for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

                unsigned int idx2 = dInds2[x];                
                for(unsigned int r=laneId; r<R; r+=32) {
                    tmp_val += vals[x] * dU2[idx2 * R + r]; 
                }
            }
            for(unsigned int r=laneId; r<R; r+=32) {  
                atomicAdd(&dY[fbr * R + r], tmp_val);    
                // dY[fbr * R + r] += tmp_val;  
            }   
        }
    }
}

__global__ void ttm_fbrLevelPar_kernel(DTYPE * vals, ITYPE *dInds2, ITYPE *fbrPtr0,
    ITYPE *fbrPtr1, unsigned int nFibers, DTYPE *dY, DTYPE *dU2, ITYPE   mode, ITYPE R, ITYPE warpPerFiber, int logOfWPF){

    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 31;
    unsigned int gId = (blockIdx.x * blockDim.x + tId);
    unsigned int workId = (tId & ((1 << (5 + logOfWPF)) - 1)) >> 5;  
    unsigned int fbr = gId >> (5 + logOfWPF); // 5: minimum 1 WARP (2^5) 
    DTYPE tmp_val;

    if(fbr < nFibers){ 
                                      
        tmp_val = 0;
        for(unsigned int x = fbrPtr1[fbr] + workId; x < fbrPtr1[fbr+1]; x+=warpPerFiber) {

            unsigned int idx2 = dInds2[x];                
            for(unsigned int r=laneId; r<R; r+=32) {
                tmp_val += vals[x] * dU2[idx2 * R + r]; 
            }
        }
        for(unsigned int r=laneId; r<R; r+=32) {  
            atomicAdd(&dY[fbr * R + r], tmp_val);    
            // dY[fbr * R + r] += tmp_val;  
        }   
        // }
    }
}

int TTM_GPU(Tensor &X, semiSpTensor &Y, Matrix *U, const Options &Opt){
    //allocate and memcpy GPU memory
    if(Opt.verbose)
        cout << "FIX fiber idx" << endl;
    //Tensor
    ITYPE *dInds2, *dInds3, *dfbrPtr0, *dfbrIdx0, *dfbrPtr1, *dfbrIdx1, *dFbrPtr2, *dFbrIdx2, *dSlcMapperBin;
    DTYPE *dVals;
    int logOfWarpPerSlice = log2(Opt.warpPerSlice);
    int TbPerSlc = 1;
    int logOfTPS = log2(TbPerSlc);

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];

    // dummy bin mapper to be compatible with bin mapper when bin are not used
    X.slcMapperBin.push_back(std::vector<ITYPE>());      
    for (ITYPE s = 0; s < X.fbrIdx[0].size(); ++s)
        X.slcMapperBin[0].push_back(s);

    /* copy tensor metadata */
    checkCuda(cudaMalloc((void**) &dVals, X.totNnz * sizeof(DTYPE)), 0);
    checkCuda(cudaMalloc((void**) &dSlcMapperBin, X.slcMapperBin[0].size() * sizeof(ITYPE)), 0);
    checkCuda(cudaMalloc((void**) &dfbrIdx0, X.fbrIdx[0].size() * sizeof(ITYPE)), 0);
    checkCuda(cudaMalloc((void**) &dfbrPtr0, X.fbrPtr[0].size() * sizeof(ITYPE)), 0);
    checkCuda(cudaMalloc((void**) &dfbrPtr1, X.fbrPtr[1].size() * sizeof(ITYPE)), 0);
    checkCuda(cudaMalloc((void**) &dfbrIdx1, X.fbrIdx[1].size() * sizeof(ITYPE)), 0);

    checkCuda(cudaMemcpy(dVals, &(X.vals[0]), X.totNnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dSlcMapperBin, &(X.slcMapperBin[0][0]), X.slcMapperBin[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrPtr0, &(X.fbrPtr[0][0]), X.fbrPtr[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrIdx0, &(X.fbrIdx[0][0]), X.fbrIdx[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrPtr1, &(X.fbrPtr[1][0]), X.fbrPtr[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrIdx1, &(X.fbrIdx[1][0]), X.fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);

    /* copy matrices */
    DTYPE *dU0, *dU1, *dU2, *dU3;   
    checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);
    checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
    
   /* copy interim tensor Y */
    DTYPE *dY;   
    checkCuda(cudaMalloc((void**) &dY, Y.nRows * Y.nCols  * sizeof(DTYPE)), 0);
    cudaMemset(dY, 0, Y.nRows * Y.nCols * sizeof(DTYPE));

    if(X.ndims == 3){
        checkCuda(cudaMalloc((void**) &dInds2, X.totNnz * sizeof(ITYPE)), 0);
        checkCuda(cudaMemcpy(dInds2, &(X.inds[mode2][0]), X.totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
        
    }

    if(X.ndims == 4){
        ITYPE mode3 = X.modeOrder[3];
        checkCuda(cudaMalloc((void**) &dFbrIdx2, X.fbrIdx[2].size() * sizeof(ITYPE)), 0);
        checkCuda(cudaMalloc((void**) &dFbrPtr2, X.fbrPtr[2].size() * sizeof(ITYPE)), 0);
        checkCuda(cudaMalloc((void**) &dInds3, X.totNnz * sizeof(ITYPE)), 0);
        checkCuda(cudaMalloc((void**) &dU3, U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE)), 0);
        
        checkCuda(cudaMemcpy(dFbrPtr2, &(X.fbrPtr[2][0]), X.fbrPtr[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
        checkCuda(cudaMemcpy(dFbrIdx2, &(X.fbrIdx[2][0]), X.fbrIdx[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
        checkCuda(cudaMemcpy(dInds3, &(X.inds[mode3][0]), X.totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
        checkCuda(cudaMemcpy(dU3, &(U[mode3].vals[0]), U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
    }

    // BLOCK and GRID
    int BLOCKSIZE = 512;
    block.x = BLOCKSIZE;

    unsigned int warpPerFiber = 16;
    int logOfWarpPerFiber = log2(warpPerFiber);

    if(Opt.warpPerSlice * 32 > BLOCKSIZE){
        cout << "BLOCKSIZE is smaller than work per slice! Increase BLOCKSIZE." << endl;
        exit(0);
    }

    if(Opt.impType == 2)
        grid.x = (Opt.warpPerSlice * 32 * X.slcMapperBin[0].size() + BLOCKSIZE - 1) / BLOCKSIZE;
    else if(Opt.impType == 3)
        grid.x = (warpPerFiber * 32 * X.nFibers + BLOCKSIZE - 1) / BLOCKSIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mili = 0;


    checkCuda(cudaEventRecord(start), __LINE__);

    if(Opt.impType == 2)
        ttm_kernel<<<grid, block, 32 * sizeof(DTYPE)>>>(dVals, dfbrIdx0, dSlcMapperBin, dInds2, dfbrPtr0, dfbrPtr1,
        X.fbrIdx[0].size(), dY, dU2, Opt.mode, Opt.R, Opt.warpPerSlice, logOfWarpPerSlice); 

    /* no notion of slices, launch as many warps as nFibers */

    else if(Opt.impType == 3)
        ttm_fbrLevelPar_kernel<<<grid, block, 32 * sizeof(DTYPE)>>>(dVals, dInds2, dfbrPtr0, dfbrPtr1, 
        X.nFibers, dY, dU2, Opt.mode, Opt.R, warpPerFiber, logOfWarpPerFiber); 

    checkCuda(cudaEventRecord(stop), __LINE__);
    cudaEventSynchronize(stop);
    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
    cudaDeviceSynchronize();
    cout << "TTM GPU - time " << mili << "ms"<< endl;

    // check correctness
    checkCuda(cudaMemcpy(&Y.vals[0], dY, Y.nRows * Y.nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
    cudaFree(dVals);  cudaFree(dY);
    cudaFree(dU0); cudaFree(dU1); cudaFree(dU2); cudaFree(dU3);
    cudaFree(dInds2); cudaFree(dInds3); 
    cudaFree(dfbrIdx0); cudaFree(dfbrIdx1); cudaFree(dFbrIdx2);
    cudaFree(dfbrPtr0); cudaFree(dfbrPtr1); cudaFree(dFbrPtr2);

    return 0;
}

