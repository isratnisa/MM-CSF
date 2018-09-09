#ifndef GPU_H
#define GPU_H

	// ** Hard coded is faster
// unsigned int workId = (tId & 127) >> 5;  
// unsigned int slc = gId >> 7;
// unsigned int shRow = slc >> rowInATB;

inline cudaError_t checkCuda(cudaError_t result, int s){

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error in line : %s - %d\n", cudaGetErrorString(result), s);
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA kernel call to do COO MTTKRP 
__global__ void mttkrp_COO_kernel(DTYPE *vals, ITYPE *dInds0, ITYPE *dInds1, ITYPE *dInds2,  ITYPE nnz,
	DTYPE *dU0, DTYPE *dU1, DTYPE *dU2, ITYPE	mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int x = gId >> 5;
	
	if(x < nnz){
        DTYPE tmp_val = 0;
        ITYPE idx0 = dInds0[x];
        ITYPE idx1 = dInds1[x];
        ITYPE idx2 = dInds2[x];

        for(ITYPE r=laneId; r<R; r+=32) {           
            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r];
            atomicAdd(&dU0[idx0 * R + r], tmp_val);
        }    
	}
}

//no atomics because all 1 in HYB - COO 
__global__ void mttkrp_HYB_COO_kernel(DTYPE *vals, ITYPE *dInds0, ITYPE *dInds1, ITYPE *dInds2,  ITYPE nnz,
	DTYPE *dU0, DTYPE *dU1, DTYPE *dU2, ITYPE	mode, ITYPE R){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int x = gId >> 5;
	
	if(x < nnz){
        DTYPE tmp_val = 0;
        ITYPE idx0 = dInds0[x];
        ITYPE idx1 = dInds1[x];
        ITYPE idx2 = dInds2[x];

        for(ITYPE r=laneId; r<R; r+=32) {           
            tmp_val = vals[x] * dU1[idx1 * R + r] * dU2[idx2 * R + r];
            dU0[idx0 * R + r] += tmp_val;
        }    
	}
}

__global__ void mttkrp_CSL_kernel(DTYPE * vals, ITYPE *dSlcInds, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *slicePtr,
	ITYPE *dInds1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	DTYPE tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = slc;//dSlcMapperBin[slc];
		unsigned int idx0 = dSlcInds[mappedSlc]; 
    	int fb_st = slicePtr[mappedSlc];
		int fb_end = slicePtr[mappedSlc+1];
		tmp_val = 0;
		
		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			
		    unsigned int idx1 = dInds1[fbr];
	        unsigned int idx2 = dInds2[fbr];                
            for(unsigned int r=laneId; r<R; r+=32) {
                tmp_val += vals[fbr] * dU2[idx2 * R + r] * dU1[idx1 * R + r]; 
            }   
		}
		for(unsigned int r=laneId; r<R; r+=32) {  
			atomicAdd(&dU0[idx0 * R + r], tmp_val);    
		}
	}
}

__global__ void mttkrp_CSL_kernel_bin(DTYPE * vals, ITYPE *dSlcInds, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *slicePtr,
	ITYPE *dInds1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	DTYPE tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dSlcInds[mappedSlc]; 
    	int fb_st = slicePtr[mappedSlc];
		int fb_end = slicePtr[mappedSlc+1];
		tmp_val = 0;
		
		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			
		    unsigned int idx1 = dInds1[fbr];
	        unsigned int idx2 = dInds2[fbr];                
            for(unsigned int r=laneId; r<R; r+=32) {
                tmp_val += vals[fbr] * dU2[idx2 * R + r] * dU1[idx1 * R + r]; 
            }   
		}
		for(unsigned int r=laneId; r<R; r+=32) {  
			atomicAdd(&dU0[idx0 * R + r], tmp_val);    
		}
	}
}


// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_CSL_kernel_hvyBin(DTYPE * vals, ITYPE *dSlcInds, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *slicePtr,
	ITYPE *dInds1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
	
	unsigned int laneId = threadIdx.x & 31;
	unsigned int workId = threadIdx.x >> 5;
	unsigned int slc = blockIdx.x >> logOfTPS;
	unsigned int localBId = blockIdx.x & (TbPerSlc -1);
	
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dSlcInds[mappedSlc] ;//slc;
		unsigned int nFbr = slicePtr[mappedSlc+1] - slicePtr[mappedSlc];		
		unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS; 
		unsigned int fb_st = slicePtr[mappedSlc] + localBId * fbrPerTb ;
		unsigned int fb_end = slicePtr[mappedSlc] + (localBId + 1) * fbrPerTb ;

		tmp_val = 0;
		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			
		    unsigned int idx1 = dInds1[fbr];
	        unsigned int idx2 = dInds2[fbr];                
            for(unsigned int r=laneId; r<R; r+=32) {
                tmp_val += vals[fbr] * dU2[idx2 * R + r] * dU1[idx1 * R + r]; 
            }   
		}
		for(unsigned int r=laneId; r<R; r+=32) {  
			atomicAdd(&dU0[idx0 * R + r], tmp_val);    
		} 
	}
}

// HCSR MTTKRP : 16 WARP = 1 TB per slice
__global__ void mttkrp_HCSR_kernel_16WARP(DTYPE * vals, ITYPE *dSlcInds, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *slicePtr,
	ITYPE *fiberPtr, ITYPE *fiberIdx, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = tId >> 5; //(tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = blockIdx.x ;//gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	DTYPE tmp = 0; 
	DTYPE tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dSlcInds[mappedSlc] ;//slc;
    	int fb_st = slicePtr[mappedSlc];
		int fb_end = slicePtr[mappedSlc+1];

		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			tmp_val = 0;
	        for(unsigned int x = fiberPtr[fbr]; x < fiberPtr[fbr+1]; ++x) {

		        unsigned int idx2 = dInds2[x];                
	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU2[idx2 * R + r]; 
	            }
	        }
	        // unsigned int idx1 = dInds1[fiberPtr[fbr]]; 
	        unsigned int idx1 = fiberIdx[fbr];   
	        for(unsigned int r=laneId; r<R; r+=32) {  
	        	tmp += tmp_val * dU1[idx1 * R + r] ;     
	        }    
		}

		for(unsigned int r=laneId; r<R; r+=32) {  
            atomicAdd(&dU0[idx0 * R + r], tmp);
        } 
	}
}
// CUDA kernel call to do HCSR MTTKRP for the first bin 1 WARP per slice
__global__ void mttkrp_HCSR_kernel_COO(DTYPE * vals, ITYPE *dSlcInds, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *slicePtr,
	ITYPE *fiberPtr, ITYPE *fiberIdx, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int slc = gId >> 5; // 5: minimum 1 WARP (2^5) 
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){ 	    

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dSlcInds[mappedSlc] ;//slc;
    	int fb_st = slicePtr[mappedSlc];
		int fb_end = slicePtr[mappedSlc+1];

		for (int fbr = fb_st; fbr < fb_end; fbr++){
			tmp_val = 0;
	        for(unsigned int x = fiberPtr[fbr]; x < fiberPtr[fbr+1]; ++x) {

		        unsigned int idx2 = dInds2[x];                
	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU2[idx2 * R + r]; 
	            }
	        }
	        unsigned int idx1 = fiberIdx[fbr];
	        for(unsigned int r=laneId; r<R; r+=32) {  
	        	dU0[idx0 * R + r] += tmp_val * dU1[idx1 * R + r] ;     
	        }    
		}
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_HCSR_kernel_smllBin(DTYPE * vals, ITYPE *dSlcInds, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *slicePtr,
	ITYPE *fiberPtr, ITYPE *fiberIdx, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 31;
	unsigned int gId = (blockIdx.x * blockDim.x + tId);
	unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  
	unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) 
	// unsigned int slcPerTb = 16/warpPerSlice;
	// unsigned int shSlc = slc & slcPerTb;
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){ 	    

		// extern __shared__ DTYPE shared[]; // R
		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dSlcInds[mappedSlc] ;//slc;
    	int fb_st = slicePtr[mappedSlc];
		int fb_end = slicePtr[mappedSlc+1];

		for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
			tmp_val = 0;
	        for(unsigned int x = fiberPtr[fbr]; x < fiberPtr[fbr+1]; ++x) {

		        unsigned int idx2 = dInds2[x];                
	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU2[idx2 * R + r]; 
	                // shared[r] = 0; // move
	            }
	        }
	        unsigned int idx1 = fiberIdx[fbr];// dInds1[fiberPtr[fbr]];    
	        for(unsigned int r=laneId; r<R; r+=32) {  
	        	tmp += tmp_val * dU1[idx1 * R + r] ;     
	        }    
		}
		// __syncthreads();

		for(unsigned int r=laneId; r<R; r+=32) {  
			atomicAdd(&dU0[idx0 * R + r], tmp);
			// // atomicAdd(&shared[shSlc * R + r], tmp);
			// __syncthreads();

			//  if(workId == 0){
   			//  		for(unsigned int r=laneId; r<R; r+=32) 
			//  //dU0[idx0 * R + r] += shared[r];
			//  if(laneId == 0)
			//  	printf("GPU %d %f %f %f\n", mappedSlc, shared[shSlc * R + r], tmp, dU0[idx0 * R + r] );
			// 	}        
		}
	}
}

// CUDA kernel call to do HCSR MTTKRP 
__global__ void mttkrp_HCSR_kernel_hvyBin(DTYPE * vals, ITYPE *dSlcInds, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *slicePtr,
	ITYPE *fiberPtr, ITYPE *fiberIdx, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, 
	ITYPE	mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
	
	unsigned int laneId = threadIdx.x & 31;
	unsigned int workId = threadIdx.x >> 5;
	unsigned int slc = blockIdx.x >> logOfTPS;
	unsigned int localBId = blockIdx.x & (TbPerSlc -1);
	
	DTYPE tmp = 0, tmp_val;
		              	              
	if(slc < nSlices){

		unsigned int mappedSlc = dSlcMapperBin[slc];
		unsigned int idx0 = dSlcInds[mappedSlc] ;//slc;
		unsigned int nFbr = slicePtr[mappedSlc+1] - slicePtr[mappedSlc];		
		unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS; 
		unsigned int fb_st = slicePtr[mappedSlc] + localBId * fbrPerTb ;
		unsigned int fb_end = slicePtr[mappedSlc] + (localBId + 1) * fbrPerTb ;

		for (int fbr = fb_st + workId; fbr < fb_end && fbr < slicePtr[mappedSlc+1] ; fbr+=warpPerSlice){
			tmp_val = 0;
	        
	        for(unsigned int x = fiberPtr[fbr]; x < fiberPtr[fbr+1]; ++x) {

		        unsigned int idx2 = dInds2[x];                
	            for(unsigned int r=laneId; r<R; r+=32) {
	                tmp_val += vals[x] * dU2[idx2 * R + r]; 
	            }
	        }
	        unsigned int idx1 = fiberIdx[fbr];//dInds1[fiberPtr[fbr]];    
	        for(unsigned int r=laneId; r<R; r+=32) {  
	        	tmp += tmp_val * dU1[idx1 * R + r] ;     
	            // // atomicAdd(&dU0[idx0 * R + r], tmp);
	        }    
		}
		for(unsigned int r=laneId; r<R; r+=32) {  
            atomicAdd(&dU0[idx0 * R + r], tmp);
        } 
	}
}

int MTTKRP_COO_GPU(const Tensor &X, Matrix *U, const Options Opt){
	//allocate and memcpy GPU memory

	//Tensor
	ITYPE mode = Opt.mode;
	ITYPE R = Opt.R;
	ITYPE *dInds0, *dInds1, *dInds2;
	DTYPE *dVals;

	ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];

	checkCuda(cudaMalloc((void**) &dVals, X.nnz * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds0, X.nnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds1, X.nnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds2, X.nnz * sizeof(ITYPE)), 0);

	checkCuda(cudaMemcpy(dVals, &(X.vals[0]), X.nnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dInds0, &(X.inds[mode0][0]), X.nnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dInds1, &(X.inds[mode1][0]), X.nnz * sizeof(ITYPE) ,cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dInds2, &(X.inds[mode2][0]), X.nnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);

	// //Matrices
	DTYPE *dU0, *dU1, *dU2;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	
	// BLOCK and GRID
	int BLOCKSIZE = 128;
	dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);
	grid.x = (32 * X.nnz + BLOCKSIZE - 1) / BLOCKSIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mili = 0;

	// CUDA call
	checkCuda(cudaEventRecord(start), __LINE__);
	mttkrp_COO_kernel<<<grid, block>>>(dVals, dInds0, dInds1, dInds2, X.nnz, dU0, dU1, dU2,
								mode, R); 
	checkCuda(cudaEventRecord(stop), __LINE__);
    cudaEventSynchronize(stop);
    //cudaDeviceSynchronize();
    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
    cudaDeviceSynchronize();
    cout << "COO GPU - time " << mili << "ms"<< endl;

	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
	// print_output(U, 0);
	cudaFree(dVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2);
	cudaFree(dInds0); cudaFree(dInds1); cudaFree(dInds2);


	return 0;
}
int MTTKRP_HCSR_GPU(Tensor &X, Matrix *U, const Options &Opt){
	//allocate and memcpy GPU memory

	//Tensor
	ITYPE *dSlcInds, *dInds1, *dInds2, *dSlcPtr, *dFbrPtr, *dFbrIdx, *dSlcMapperBin;
	DTYPE *dVals;
	int logOfWarpPerSlice = log2(Opt.warpPerSlice);
	int TbPerSlc = 1;
	int logOfTPS = log2(TbPerSlc);

	ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];

	// dummy bin mapper to be compatible with bin mapper when bin are not used
	X.slcMapperBin.push_back(std::vector<ITYPE>());      
	for (int s = 0; s < X.sliceIdx.size(); ++s)
		X.slcMapperBin[0].push_back(s);

	checkCuda(cudaMalloc((void**) &dVals, X.nnz * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds2, X.nnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dSlcInds, X.sliceIdx.size() * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dSlcPtr, X.slicePtr.size() * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dSlcMapperBin, X.slcMapperBin[0].size() * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dFbrPtr, X.fiberPtr.size() * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dFbrIdx, X.fiberIdx.size() * sizeof(ITYPE)), 0);

	checkCuda(cudaMemcpy(dVals, &(X.vals[0]), X.nnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dInds2, &(X.inds[mode2][0]), X.nnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dSlcPtr, &(X.slicePtr[0]), X.slicePtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dSlcInds, &(X.sliceIdx[0]), X.sliceIdx.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dSlcMapperBin, &(X.slcMapperBin[0][0]), X.slcMapperBin[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dFbrPtr, &(X.fiberPtr[0]), X.fiberPtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dFbrIdx, &(X.fiberIdx[0]), X.fiberIdx.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);

	// //Matrices
	DTYPE *dU0, *dU1, *dU2;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	
	// BLOCK and GRID
	int BLOCKSIZE = 512;

	if(Opt.warpPerSlice * 32 > BLOCKSIZE){
		cout << "BLOCKSIZE is smaller than work per slice! Increase BLOCKSIZE." << endl;
		exit(0);
	}

	dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);
	grid.x = (Opt.warpPerSlice * 32 * X.dims[mode0] + BLOCKSIZE - 1) / BLOCKSIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mili = 0;

	checkCuda(cudaEventRecord(start), __LINE__);

	mttkrp_HCSR_kernel_COO<<<grid, block, 32 * sizeof(DTYPE)>>>(dVals, dSlcInds, dSlcMapperBin, dInds2, dSlcPtr, dFbrPtr, dFbrIdx,
		X.sliceIdx.size(), dU0, dU1, dU2,Opt.mode, Opt.R, Opt.warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 

	// mttkrp_HCSR_kernel_smllBin<<<grid, block, 32 * sizeof(DTYPE)>>>(dVals, dSlcInds, dSlcMapperBin, dInds2, dSlcPtr, dFbrPtr, dFbrIdx,
	// 	X.sliceIdx.size(), dU0, dU1, dU2,Opt.mode, Opt.R, Opt.warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 

	checkCuda(cudaEventRecord(stop), __LINE__);
    cudaEventSynchronize(stop);
    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
    cudaDeviceSynchronize();
    cout << "HCSR GPU - time " << mili << "ms"<< endl;

	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
	cudaFree(dVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2);
	cudaFree(dSlcInds); cudaFree(dInds1); cudaFree(dInds2); cudaFree(dSlcPtr); cudaFree(dFbrPtr);

	return 0;
}
int MTTKRP_TILED_COO_GPU(TiledTensor *TiledX, Matrix *U, const Options Opt){
	//allocate and memcpy GPU memory

	//Tensor
	ITYPE mode = Opt.mode;
	ITYPE R = Opt.R;
	ITYPE *dInds0, *dInds1, *dInds2;
	ITYPE dLoc = 0, totNnz = 0;
	DTYPE *dVals;

	// All tile same mode
	ITYPE mode0 = TiledX[0].modeOrder[0];
    ITYPE mode1 = TiledX[0].modeOrder[1];
    ITYPE mode2 = TiledX[0].modeOrder[2];

	for (int tile = 0; tile < Opt.nTile; ++tile)
		totNnz += TiledX[tile].nnz;

	checkCuda(cudaMalloc((void**) &dVals, totNnz * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds0, totNnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds1, totNnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds2, totNnz * sizeof(ITYPE)), 0);

	for (int tile = 0; tile < Opt.nTile; ++tile){
		
		if(tile > 0) 
			dLoc += TiledX[tile-1].nnz;

		checkCuda(cudaMemcpy(dVals + dLoc, &(TiledX[tile].vals[0]), TiledX[tile].nnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dInds0 + dLoc, &(TiledX[tile].inds[mode0][0]), TiledX[tile].nnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dInds1 + dLoc, &(TiledX[tile].inds[mode1][0]), TiledX[tile].nnz * sizeof(ITYPE) ,cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dInds2 + dLoc, &(TiledX[tile].inds[mode2][0]), TiledX[tile].nnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	}

	// //Matrices
	DTYPE *dU0, *dU1, *dU2;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	
	// BLOCK and GRID
	int BLOCKSIZE = 128;
	dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);
	
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mili = 0, GPUTime = 0;

	// CUDA call
	dLoc = 0;
	for (int tile = 0; tile < Opt.nTile; ++tile){
		
		if(tile > 0) 
			dLoc += TiledX[tile-1].nnz;

		cout << "Tile " << tile << " launched.. "<<endl;
		
		grid.x = (32 * TiledX[tile].nnz + BLOCKSIZE - 1) / BLOCKSIZE;

		checkCuda(cudaEventRecord(start), __LINE__);
		mttkrp_COO_kernel<<<grid, block>>>(dVals + dLoc, dInds0 + dLoc, dInds1 + dLoc, dInds2 + dLoc, TiledX[tile].nnz, dU0, dU1, dU2,
								mode, R); 
	
		checkCuda(cudaEventRecord(stop), __LINE__);
	    cudaEventSynchronize(stop);
	    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
	    cudaDeviceSynchronize();
	    cout << "Tile: " << tile << " - time " << mili << "ms"<< endl;
	    GPUTime += mili;
	   
	}
	cout << "COO GPU - time " << GPUTime << "ms"<< endl;

	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
	cudaFree(dVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2);
	cudaFree(dInds0); cudaFree(dInds1); cudaFree(dInds2);

	return 0;
}
int MTTKRP_TILED_HCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt){
	//allocate and memcpy GPU memory

	//Tensor
	ITYPE *dSlcInds, *dInds1, *dInds2, *dSlcPtr, *dFbrPtr, *dFbrIdx, *dSlcMapperBin;
	DTYPE *dVals;
	ITYPE dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0, dFbrLoc =0, dBinLoc = 0;
	ITYPE totNnz = 0, totSlcPtr = 0, totSlcIdx = 0, totFbrPtr = 0;
	int warpPerSlice = Opt.warpPerSlice;
	int logOfWarpPerSlice = log2(Opt.warpPerSlice);
	int TbPerSlc = 1;
	int logOfTPS = log2(TbPerSlc);

	// All tile same mode
	ITYPE mode0 = TiledX[0].modeOrder[0];
    ITYPE mode1 = TiledX[0].modeOrder[1];
    ITYPE mode2 = TiledX[0].modeOrder[2];

	for (int tile = 0; tile < Opt.nTile; ++tile){
		totNnz += TiledX[tile].nnz;
		totSlcPtr += TiledX[tile].slicePtr.size() ;
		totSlcIdx += TiledX[tile].sliceIdx.size() ;
		totFbrPtr += TiledX[tile].fiberPtr.size() ;
	}

	checkCuda(cudaMalloc((void**) &dVals, totNnz * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dInds2, totNnz * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dSlcPtr, totSlcPtr * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dSlcInds, totSlcIdx * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dSlcMapperBin, totSlcPtr * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dFbrPtr, totFbrPtr * sizeof(ITYPE)), 0);
	checkCuda(cudaMalloc((void**) &dFbrIdx, totFbrPtr * sizeof(ITYPE)), 0);

	for (int tile = 0; tile < Opt.nTile; ++tile){	
		if(tile > 0) {
			dLoc += TiledX[tile-1].nnz;
			dSlcLoc += TiledX[tile - 1].slicePtr.size(); // all tile same
			dSlcIdxLoc += TiledX[tile - 1].sliceIdx.size(); 
			dFbrLoc += TiledX[tile - 1].fiberPtr.size();
		}

		checkCuda(cudaMemcpy(dVals + dLoc, &(TiledX[tile].vals[0]), TiledX[tile].nnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dInds2 + dLoc, &(TiledX[tile].inds[mode2][0]), TiledX[tile].nnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dSlcPtr + dSlcLoc, &(TiledX[tile].slicePtr[0]), TiledX[tile].slicePtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dSlcInds + dSlcIdxLoc, &(TiledX[tile].sliceIdx[0]), TiledX[tile].sliceIdx.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dFbrPtr + dFbrLoc, &(TiledX[tile].fiberPtr[0]), TiledX[tile].fiberPtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dFbrIdx + dFbrLoc, &(TiledX[tile].fiberIdx[0]), TiledX[tile].fiberPtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	
		dBinLoc = 0;
		for (int bin = 0; bin < Opt.nBin; ++bin){

			if(bin > 0)
				dBinLoc += TiledX[tile].slcMapperBin[bin-1].size();

		    checkCuda(cudaMemcpy(dSlcMapperBin + dSlcIdxLoc + dBinLoc, &(TiledX[tile].slcMapperBin[bin][0]), TiledX[tile].slcMapperBin[bin].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		}
	}

	// //Matrices
	DTYPE *dU0, *dU1, *dU2;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	
	// BLOCK and GRID
	int BLOCKSIZE = 512;
	unsigned int rowInATB = BLOCKSIZE / (Opt.warpPerSlice*32); 

	if(Opt.warpPerSlice * 32 > BLOCKSIZE){
		cout << "BLOCKSIZE is smaller than work per slice! Increase BLOCKSIZE." << endl;
		exit(0);
	}

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t streams[Opt.nBin];
    float mili = 0, GPUTime = 0, CPUtimer = 0;

	dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0; dFbrLoc =0;

	for (int bin = 0; bin < Opt.nBin; ++bin)
		cudaStreamCreate(&streams[bin]);

	for (int tile = 0; tile < Opt.nTile; ++tile){

		dBinLoc = 0;
		
		if(tile > 0) {
			dLoc += TiledX[tile-1].nnz;
			dSlcLoc += TiledX[tile - 1].slicePtr.size(); 
			dSlcIdxLoc += TiledX[tile - 1].sliceIdx.size(); 
			dFbrLoc += TiledX[tile - 1].fiberPtr.size();
		}

		BLOCKSIZE = 512;
		dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

		int smallBinEndsAt = 5;
		int slcPerTb = 0;

		// Process small bins.. accepts 2 slice 1 TB

		double t0 = seconds();
		checkCuda(cudaEventRecord(start), __LINE__);
		
		for (int bin = 0; bin < Opt.nBin ; ++bin){
							
			if(bin == -1){

				warpPerSlice = 1;

				logOfWarpPerSlice = 0;

				grid.x = ( 32 * TiledX[tile].slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

				mttkrp_HCSR_kernel_COO<<<grid, block, 0, streams[bin]>>>(dVals + dLoc, dSlcInds + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
					dInds2 + dLoc, dSlcPtr + dSlcLoc, dFbrPtr + dFbrLoc,  dFbrIdx + dFbrLoc, TiledX[tile].slcMapperBin[bin].size(), 
					dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 
			}

			else if(bin > -1 && bin < smallBinEndsAt){

				TbPerSlc = 1;

				warpPerSlice = ((bin > 0) ? 2 << (bin - 1) : 1);

				if(warpPerSlice > 16)		
					warpPerSlice = 16;
				logOfWarpPerSlice = log2(warpPerSlice);
				slcPerTb = 16 / warpPerSlice;

				ITYPE shSize = 0;//slcPerTb * 32 * sizeof(DTYPE);

				dBinLoc += ((bin > 0) ? TiledX[tile].slcMapperBin[bin-1].size() : 0);

				grid.x = ( TbPerSlc * warpPerSlice * 32 * TiledX[tile].slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

				mttkrp_HCSR_kernel_smllBin<<<grid, block, shSize , streams[bin]>>>(dVals + dLoc, dSlcInds + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
					dInds2 + dLoc, dSlcPtr + dSlcLoc, dFbrPtr + dFbrLoc,  dFbrIdx + dFbrLoc, TiledX[tile].slcMapperBin[bin].size(), 
					dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 
			}
			
			// Processing heavy bin.. multiple TB per slice
			else{

				TbPerSlc = 1 << (bin - smallBinEndsAt + 1); // 1st big bin starts with 1 TB 1 << 1 not 1 << 5
				if(TbPerSlc > 32) TbPerSlc = 32;		
				logOfTPS = log2(TbPerSlc);

				warpPerSlice = 16;
				logOfWarpPerSlice = 4;

				dBinLoc += TiledX[tile].slcMapperBin[bin-1].size();
						
				grid.x = (TbPerSlc * warpPerSlice * 32 * TiledX[tile].slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;
				
				mttkrp_HCSR_kernel_hvyBin<<<grid, block, 0, streams[bin]>>>(dVals + dLoc, dSlcInds + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
					dInds2 + dLoc, dSlcPtr + dSlcLoc, dFbrPtr + dFbrLoc,  dFbrIdx + dFbrLoc, TiledX[tile].slcMapperBin[bin].size(), 
					dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice,  TbPerSlc, logOfTPS); 

			}

		}
		checkCuda(cudaEventRecord(stop), __LINE__);
	    cudaEventSynchronize(stop);
	    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
	    CPUtimer += seconds() - t0;
	    cudaDeviceSynchronize();
	    GPUTime += mili;
	    
	    if(Opt.verbose){
	    	cout << "Tile: " << tile << " - time: " << mili << "ms";
	    	cout <<" nnz: " << TiledX[tile].nnz << " nFibers: "
	    	<< TiledX[tile].fiberPtr.size() << " nSlc " << TiledX[tile].sliceIdx.size() << " ";
			cout << endl;
		}   
	}
	cout << "HCSR GPU: " << GPUTime << endl;

	for (int bin = 0; bin < Opt.nBin; ++bin)
		cudaStreamDestroy(streams[bin]);
	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
	cudaFree(dVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2);
	cudaFree(dSlcInds); cudaFree(dInds1); cudaFree(dInds2); cudaFree(dSlcPtr); cudaFree(dFbrPtr);

	return 0;
}

int MTTKRP_HYB_GPU(const HYBTensor &HybX, Matrix *U, const Options &Opt){
	//allocate and memcpy GPU memory

	//Tensor
	ITYPE *dCOOInds0, *dCOOInds1, *dCOOInds2;
	ITYPE *dCSLSlcPtr, *dCSLSlcInds, *dCSLInds1, *dCSLInds2, *dCSLSlcMapperBin;
	ITYPE *dSlcPtr, *dSlcInds, *dInds1, *dInds2, *dFbrPtr, *dFbrIdx, *dSlcMapperBin;

	DTYPE *dVals, *dCOOVals, *dCSLVals;
	ITYPE dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0, dFbrLoc =0, dBinLoc = 0, dCSLBinLoc = 0;
	ITYPE totNnz = 0, totSlcPtr = 0, totSlcIdx = 0, totFbrPtr = 0;
	int warpPerSlice = Opt.warpPerSlice;
	int logOfWarpPerSlice = log2(Opt.warpPerSlice);
	int TbPerSlc = 1;
	int logOfTPS = log2(TbPerSlc);

	// All tile same mode
	ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];

    // ****** mem op HYB COO *******
    if(HybX.COOnnz > 0){
		
		checkCuda(cudaMalloc((void**) &dCOOVals, HybX.COOnnz * sizeof(DTYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCOOInds0, HybX.COOnnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCOOInds1, HybX.COOnnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCOOInds2, HybX.COOnnz * sizeof(ITYPE)), 0);

		checkCuda(cudaMemcpy(dCOOVals, &(HybX.COOvals[0]), HybX.COOnnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCOOInds0, &(HybX.COOinds[mode0][0]), HybX.COOnnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCOOInds1, &(HybX.COOinds[mode1][0]), HybX.COOnnz * sizeof(ITYPE) ,cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCOOInds2, &(HybX.COOinds[mode2][0]), HybX.COOnnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
	}

   // ****** mem op HYB CSL *******

	if(HybX.CSLnnz > 0){

		checkCuda(cudaMalloc((void**) &dCSLVals, HybX.CSLnnz * sizeof(DTYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLSlcPtr,  HybX.CSLslicePtr.size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLSlcInds, HybX.CSLsliceIdx.size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLInds1, HybX.CSLnnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLInds2, HybX.CSLnnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dCSLSlcMapperBin, HybX.CSLslicePtr.size() * sizeof(ITYPE)), 0);

		checkCuda(cudaMemcpy(dCSLVals, &(HybX.CSLvals[0]), HybX.CSLnnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);	
		checkCuda(cudaMemcpy(dCSLSlcPtr + dSlcLoc, &(HybX.CSLslicePtr[0]), HybX.CSLslicePtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCSLSlcInds + dSlcIdxLoc, &(HybX.CSLsliceIdx[0]), HybX.CSLsliceIdx.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCSLInds1, &(HybX.CSLinds[mode1][0]), HybX.CSLnnz * sizeof(ITYPE) ,cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dCSLInds2, &(HybX.CSLinds[mode2][0]), HybX.CSLnnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		
		dCSLBinLoc = 0;
		for (int bin = 0; bin < Opt.nBin; ++bin){

			if(bin > 0)
				dCSLBinLoc += HybX.CSLslcMapperBin[bin-1].size();

			if(HybX.CSLslcMapperBin[bin].size() > 0)
		    	checkCuda(cudaMemcpy(dCSLSlcMapperBin + dSlcIdxLoc + dCSLBinLoc, &(HybX.CSLslcMapperBin[bin][0]), HybX.CSLslcMapperBin[bin].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		}
	}
 
    // ****** mem op HYB HCSR *******

    if(HybX.nnz > 0){

		checkCuda(cudaMalloc((void**) &dVals, HybX.nnz * sizeof(DTYPE)), 0);
		checkCuda(cudaMalloc((void**) &dInds2, HybX.nnz * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dSlcPtr,  HybX.slicePtr.size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dSlcInds, HybX.sliceIdx.size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dSlcMapperBin, HybX.slicePtr.size() * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dFbrPtr, HybX.fiberPtr.size()  * sizeof(ITYPE)), 0);
		checkCuda(cudaMalloc((void**) &dFbrIdx, HybX.fiberPtr.size() * sizeof(ITYPE)), 0);

		checkCuda(cudaMemcpy(dVals + dLoc, &(HybX.vals[0]), HybX.nnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dInds2 + dLoc, &(HybX.inds[mode2][0]), HybX.nnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dSlcPtr + dSlcLoc, &(HybX.slicePtr[0]), HybX.slicePtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dSlcInds + dSlcIdxLoc, &(HybX.sliceIdx[0]), HybX.sliceIdx.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dFbrPtr + dFbrLoc, &(HybX.fiberPtr[0]), HybX.fiberPtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		checkCuda(cudaMemcpy(dFbrIdx + dFbrLoc, &(HybX.fiberIdx[0]), HybX.fiberPtr.size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);

		dBinLoc = 0;
		for (int bin = 0; bin < Opt.nBin; ++bin){

			if(bin > 0)
				dBinLoc += HybX.slcMapperBin[bin-1].size();

			if(HybX.slcMapperBin[bin].size() > 0)
		    	checkCuda(cudaMemcpy(dSlcMapperBin + dSlcIdxLoc + dBinLoc, &(HybX.slcMapperBin[bin][0]), HybX.slcMapperBin[bin].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
		}
	}
	
	// //Matrices
	DTYPE *dU0, *dU1, *dU2;	
	checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

	cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
	checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
	
	// BLOCK and GRID
	int BLOCKSIZE = 512;
	dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);
	unsigned int rowInATB = BLOCKSIZE / (Opt.warpPerSlice*32); 

	if(Opt.warpPerSlice * 32 > BLOCKSIZE){
		cout << "BLOCKSIZE is smaller than work per slice! Increase BLOCKSIZE." << endl;
		exit(0);
	}

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t streams[Opt.nBin + 1];
    float mili = 0, GPUTime = 0, CPUtimer = 0, HYBTime = 0;

	dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0; dFbrLoc =0;

	for (int bin = 0; bin < Opt.nBin + 1; ++bin)
		cudaStreamCreate(&streams[bin]);


	// ******* CUDA COO *******

	if(HybX.COOnnz > 0){

		BLOCKSIZE = 128;
		block.x = BLOCKSIZE;
		grid.x = (32 * HybX.COOnnz + BLOCKSIZE - 1) / BLOCKSIZE;

		// CUDA call
		checkCuda(cudaEventRecord(start), __LINE__);

		mttkrp_HYB_COO_kernel<<<grid, block, 0, 0>>>(dCOOVals, dCOOInds0, dCOOInds1, dCOOInds2, HybX.COOnnz, dU0, dU1, dU2,
									Opt.mode, Opt.R); 
		checkCuda(cudaEventRecord(stop), __LINE__);
	    cudaEventSynchronize(stop);
	    //cudaDeviceSynchronize();
	    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
	    cudaDeviceSynchronize();
	    HYBTime += mili;
	    if(Opt.verbose)
	    	cout << "COO GPU " << mili << "ms"<< endl;

	}
	// ******* CUDA CSL *******

	if(HybX.CSLnnz > 0){

		int smallBinEndsAt = 5;
		int slcPerTb = 0;

		BLOCKSIZE = 512;
		block.x = BLOCKSIZE;

		warpPerSlice = 2;
		logOfWarpPerSlice = log2(warpPerSlice);
		grid.x = ( TbPerSlc * warpPerSlice * 32 * HybX.CSLsliceIdx.size() + BLOCKSIZE - 1) / BLOCKSIZE;
		mili = 0; 
		dCSLBinLoc = 0;

	    checkCuda(cudaEventRecord(start), __LINE__);

		// mttkrp_CSL_kernel<<<grid, block>>>(dCSLVals, dCSLSlcInds, dSlcMapperBin, dCSLInds2, dCSLSlcPtr, 
		// 	dCSLInds1, HybX.CSLsliceIdx.size(), dU0, dU1, dU2,Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 

		// checkCuda(cudaEventRecord(stop), __LINE__);
	 //    cudaEventSynchronize(stop);
	 //    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
	 //    cudaDeviceSynchronize();
	 //    HYBTime += mili;
		
		for (int bin = 0; bin < Opt.nBin ; ++bin){

			if(bin < smallBinEndsAt){

				TbPerSlc = 1;

				warpPerSlice = ((bin > 0) ? 2 << (bin) : 1);

				if(warpPerSlice > 16)		
					warpPerSlice = 16;
				logOfWarpPerSlice = log2(warpPerSlice);
				slcPerTb = 16 / warpPerSlice;

				dCSLBinLoc += ((bin > 0) ? HybX.CSLslcMapperBin[bin-1].size() : 0);

				grid.x = ( TbPerSlc * warpPerSlice * 32 * HybX.CSLslcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

				if( HybX.CSLslcMapperBin[bin].size() > 0)
				mttkrp_CSL_kernel_bin<<<grid, block, 0, streams[bin + 1]>>>(dCSLVals, dCSLSlcInds, dCSLSlcMapperBin + dCSLBinLoc, 
					dCSLInds2, dCSLSlcPtr, dCSLInds1, HybX.CSLslcMapperBin[bin].size(), 
					dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 
			
			}
			// Processing heavy bin.. multiple TB per slice
			else{

				TbPerSlc = 1 << (bin - smallBinEndsAt + 1); // 1st big bin starts with 1 TB 1 << 1 not 1 << 5
				if(TbPerSlc > 32) TbPerSlc = 32;		
				logOfTPS = log2(TbPerSlc);

				warpPerSlice = 16;
				logOfWarpPerSlice = 4;

				dCSLBinLoc += HybX.CSLslcMapperBin[bin-1].size();
						
				grid.x = (TbPerSlc * warpPerSlice * 32 * HybX.CSLslcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;
				
				mttkrp_CSL_kernel_hvyBin<<<grid, block, 0, streams[bin]>>>(dCSLVals + dLoc, dCSLSlcInds + dSlcIdxLoc, dCSLSlcMapperBin + dSlcIdxLoc + dCSLBinLoc, 
					dCSLInds2 + dLoc, dCSLSlcPtr + dSlcLoc, dCSLInds1, HybX.CSLslcMapperBin[bin].size(), 
					dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice,  TbPerSlc, logOfTPS); 

			}

		}
		checkCuda(cudaEventRecord(stop), __LINE__);
	    cudaEventSynchronize(stop);
	    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
	    cudaDeviceSynchronize();
	    HYBTime += mili;
	    if(Opt.verbose)
	    	cout << "CSL GPU " << mili << "ms"<< endl;
	}

	// ******* CUDA HSCR *******

	if(HybX.nnz > 0){
		dBinLoc = 0;

		BLOCKSIZE = 512;
		block.x = BLOCKSIZE;

		int smallBinEndsAt = 5;
		
		int slcPerTb = 0;

		// Process small bins.. accepts 2 slice 1 TB
		mili = 0 ;
		double t0 = seconds();
		checkCuda(cudaEventRecord(start), __LINE__);
		
		for (int bin = 0; bin < Opt.nBin ; ++bin){
							
			if(bin == -1){

				warpPerSlice = 1;

				logOfWarpPerSlice = 0;

				grid.x = ( 32 * HybX.slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

				mttkrp_HCSR_kernel_COO<<<grid, block, 0, streams[bin]>>>(dVals + dLoc, dSlcInds + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
					dInds2 + dLoc, dSlcPtr + dSlcLoc, dFbrPtr + dFbrLoc,  dFbrIdx + dFbrLoc, HybX.slcMapperBin[bin].size(), 
					dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 
			}

			else if(bin > -1 && bin < smallBinEndsAt){

				TbPerSlc = 1;

				warpPerSlice = ((bin > 0) ? 2 << (bin - 1) : 1);

				if(warpPerSlice > 16)		
					warpPerSlice = 16;
				logOfWarpPerSlice = log2(warpPerSlice);
				slcPerTb = 16 / warpPerSlice;

				ITYPE shSize = 0;//slcPerTb * 32 * sizeof(DTYPE);

				dBinLoc += ((bin > 0) ? HybX.slcMapperBin[bin-1].size() : 0);

				grid.x = ( TbPerSlc * warpPerSlice * 32 * HybX.slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

				mttkrp_HCSR_kernel_smllBin<<<grid, block, shSize , streams[bin]>>>(dVals + dLoc, dSlcInds + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
					dInds2 + dLoc, dSlcPtr + dSlcLoc, dFbrPtr + dFbrLoc,  dFbrIdx + dFbrLoc, HybX.slcMapperBin[bin].size(), 
					dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice, TbPerSlc, logOfTPS); 
			}
			
			// Processing heavy bin.. multiple TB per slice
			else{

				TbPerSlc = 1 << (bin - smallBinEndsAt + 1); // 1st big bin starts with 1 TB 1 << 1 not 1 << 5
				if(TbPerSlc > 32) TbPerSlc = 32;		
				logOfTPS = log2(TbPerSlc);

				warpPerSlice = 16;
				logOfWarpPerSlice = 4;

				dBinLoc += HybX.slcMapperBin[bin-1].size();
						
				grid.x = (TbPerSlc * warpPerSlice * 32 * HybX.slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;
				
				mttkrp_HCSR_kernel_hvyBin<<<grid, block, 0, streams[bin]>>>(dVals + dLoc, dSlcInds + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
					dInds2 + dLoc, dSlcPtr + dSlcLoc, dFbrPtr + dFbrLoc,  dFbrIdx + dFbrLoc, HybX.slcMapperBin[bin].size(), 
					dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlice, logOfWarpPerSlice,  TbPerSlc, logOfTPS); 

			}

		}
		checkCuda(cudaEventRecord(stop), __LINE__);
	    cudaEventSynchronize(stop);
	    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
	    CPUtimer += seconds() - t0;
	    cudaDeviceSynchronize();

	    HYBTime += mili;
	  	if(Opt.verbose)
			cout << "HCSR GPU: " << mili << endl;
	}
	cout << "HYB GPU: " << HYBTime << endl;

	for (int bin = 0; bin < Opt.nBin; ++bin)
		cudaStreamDestroy(streams[bin]);
	// check correctness
	checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);
	cudaFree(dVals); cudaFree(dCOOVals); cudaFree(dCSLVals); 
	cudaFree(dU0); cudaFree(dU1); cudaFree(dU2);
	cudaFree(dSlcInds); cudaFree(dInds1); cudaFree(dInds2); cudaFree(dSlcPtr); cudaFree(dFbrPtr);
	cudaFree(dCSLSlcInds); cudaFree(dCSLInds1); cudaFree(dCSLInds2); cudaFree(dCSLSlcPtr); 
	cudaFree(dCOOInds0); cudaFree(dCOOInds1); cudaFree(dCOOInds2); 

	return 0;
}
#endif
