#include <iostream>
#include "ttm_cpu.h"
//implementation 1; MTTKRP on CPU using COO

int TTM_CPU(const Tensor &X, semiSpTensor &Y, Matrix *U, const Options &Opt){

    ITYPE R = Opt.R;
    ITYPE mode2 = X.modeOrder[2];
    DTYPE const * const  U2 = U[mode2].vals;
    DTYPE *tmp_val = new DTYPE[R];
       
    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
        
        for (int fbr = X.fbrPtr[0][slc]; fbr <  X.fbrPtr[0][slc+1]; ++fbr){

            memset(tmp_val, 0, R * sizeof(DTYPE));
             
            for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {

                ITYPE idx2 = X.inds[mode2][x];  
    
                for(ITYPE r=0; r<R; ++r) 
                    tmp_val[r] += X.vals[x] * U2[idx2 * R + r]; 
                
            }

            for(ITYPE r=0; r<R; ++r){ 
                Y.vals[fbr * R + r] += tmp_val[r];
            }
        }
    }
}