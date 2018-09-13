#include <iostream>
#include "mttkrp_cpu.h"
//implementation 1; MTTKRP on CPU using COO

int MTTKRP_COO_CPU(const Tensor &X, Matrix *U, const Options &Opt){

    ITYPE R = Opt.R;
    // #pragma omp parallel for //reduction(+:U[0].vals[:R])
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    
    for(ITYPE x=0; x<X.totNnz; ++x) {

        DTYPE tmp_val = 0;
        ITYPE idx0 = X.inds[mode0][x];
        ITYPE idx1 = X.inds[mode1][x];
        ITYPE idx2 = X.inds[mode2][x];
       
        // #pragma omp atomic
        for(ITYPE r=0; r<R; ++r) {            
            tmp_val = X.vals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r];
            U[mode0].vals[idx0 * R + r] += tmp_val;
        }
    }
}

int MTTKRP_HCSR_CPU(const Tensor &X, Matrix *U, const Options &Opt){
        
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE R = Opt.R;

     // MTTKRP_HCSR_CPU_RSTRCT(const Tensor &X, U[], const Options &Opt)
    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;

    // ITYPE const * const __restrict__ arrIdx2 = &(X.inds[mode2][0]);
  
    // #pragma omp parallel
    {    
        DTYPE *tmp_val = new DTYPE[R];
        DTYPE *outBuffer = new DTYPE[R];

        // #pragma omp for
        for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {
            memset(outBuffer, 0, R * sizeof(DTYPE));

            ITYPE idx0 = X.sliceIdx[slc];
            const int fb_st = X.slicePtr[slc];
            const int fb_end = X.slicePtr[slc+1];
            
            for (int fbr = fb_st; fbr < fb_end; ++fbr){
                #pragma omp simd
                for(ITYPE r=0; r<R; ++r){
                    tmp_val[r] = 0;
                }
                 
                for(ITYPE x = X.fiberPtr[fbr]; x < X.fiberPtr[fbr+1]; ++x) {

                    ITYPE idx2 = X.inds[mode2][x];  
                    // ITYPE idx2 = arrIdx2[x];  
                    #pragma omp simd              
                    for(ITYPE r=0; r<R; ++r) {
                        tmp_val[r] += X.vals[x] * U2[idx2 * R + r]; 
                    }
                }
                
                ITYPE idx1 = X.fiberIdx[fbr];
                 #pragma omp simd
                for(ITYPE r=0; r<R; ++r) 
                    outBuffer[r] += tmp_val[r] * U1[idx1 * R + r];               
            }
            for(ITYPE r=0; r<R; ++r) 
                U0[idx0 * R + r] += outBuffer[r];
        }
    }
}

int MTTKRP_HYB_HCSR_CPU(HYBTensor &X, Matrix *U, Options &Opt){

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE R = Opt.R;

     // MTTKRP_HCSR_CPU_RSTRCT(const Tensor &X, U[], const Options &Opt)
    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;

    // ITYPE const * const __restrict__ arrIdx2 = &(X.inds[mode2][0]);
  
    // #pragma omp parallel
    {    
        DTYPE *tmp_val = new DTYPE[R];
        DTYPE *outBuffer = new DTYPE[R];

        // #pragma omp for
        for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {
            memset(outBuffer, 0, R * sizeof(DTYPE));

            ITYPE idx0 = X.sliceIdx[slc];
            const int fb_st = X.slicePtr[slc];
            const int fb_end = X.slicePtr[slc+1];
            
            for (int fbr = fb_st; fbr < fb_end; ++fbr){
                #pragma omp simd
                for(ITYPE r=0; r<R; ++r){
                    tmp_val[r] = 0;
                }
                 
                for(ITYPE x = X.fiberPtr[fbr]; x < X.fiberPtr[fbr+1]; ++x) {

                    ITYPE idx2 = X.inds[mode2][x];  
                    // ITYPE idx2 = arrIdx2[x];  
                    #pragma omp simd              
                    for(ITYPE r=0; r<R; ++r) {
                        tmp_val[r] += X.vals[x] * U2[idx2 * R + r]; 
                    }
                }
                
                ITYPE idx1 = X.fiberIdx[fbr];
                 #pragma omp simd
                for(ITYPE r=0; r<R; ++r) 
                    outBuffer[r] += tmp_val[r] * U1[idx1 * R + r];               
            }
            for(ITYPE r=0; r<R; ++r) 
                U0[idx0 * R + r] += outBuffer[r];
        }
    }
}

int MTTKRP_HYB_CSL_CPU( HYBTensor &HybX, Matrix *U, Options &Opt){
    ITYPE R = Opt.R;
    // #pragma omp parallel for //reduction(+:U[0].vals[:R])
    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    
    for(ITYPE slc = 0; slc < HybX.CSLsliceIdx.size(); ++slc) {

        ITYPE idx0 = HybX.CSLsliceIdx[slc];
        const int fb_st = HybX.CSLslicePtr[slc];
        const int fb_end = HybX.CSLslicePtr[slc+1];
        
        for (int fbr = fb_st; fbr < fb_end; ++fbr){

            DTYPE tmp_val = 0;
            ITYPE idx1 = HybX.CSLinds[mode1][fbr];
            ITYPE idx2 = HybX.CSLinds[mode2][fbr];    
           
            // #pragma omp atomic
            for(ITYPE r=0; r<R; ++r) {            
                tmp_val = HybX.CSLvals[fbr] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r];
                U[mode0].vals[idx0 * R + r] += tmp_val;
            }
        }
    }
}

int MTTKRP_HYB_CPU( HYBTensor &HybX, Matrix *U, Options &Opt){

    ITYPE R = Opt.R;
    // #pragma omp parallel for //reduction(+:U[0].vals[:R])
    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    
    // COO PART
    for(ITYPE x = 0; x < HybX.COOnnz; ++x) {

        DTYPE tmp_val = 0;
        ITYPE idx0 = HybX.COOinds[mode0][x];
        ITYPE idx1 = HybX.COOinds[mode1][x];
        ITYPE idx2 = HybX.COOinds[mode2][x];
       
        // #pragma omp atomic
        for(ITYPE r=0; r<R; ++r) {            
            tmp_val = HybX.COOvals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r];
            U[mode0].vals[idx0 * R + r] += tmp_val;
        }
    }
    // CSSL part
    MTTKRP_HYB_CSL_CPU(HybX, U, Opt);
    // HCSR part
    MTTKRP_HYB_HCSR_CPU(HybX, U, Opt);

}

int MTTKRP_TILED_COO_CPU(TiledTensor *TiledX, Matrix *U, const Options &Opt){
    
    ITYPE mode = Opt.mode; 
    ITYPE R = Opt.R;

    // all tile same mode..doesnt matter
    ITYPE mode0 = TiledX[0].modeOrder[0];
    ITYPE mode1 = TiledX[0].modeOrder[1];
    ITYPE mode2 = TiledX[0].modeOrder[2];

    for (int tile = 0; tile < Opt.nTile; ++tile)
    {
        for(ITYPE x=0; x<TiledX[tile].totNnz; ++x) {

            DTYPE tmp_val = 0;
            ITYPE idx0 = TiledX[tile].inds[mode0][x];
            ITYPE idx1 = TiledX[tile].inds[mode1][x];
            ITYPE idx2 = TiledX[tile].inds[mode2][x];
            
            for(ITYPE r=0; r<R; ++r) {            
                tmp_val = TiledX[tile].vals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r];
                U[mode0].vals[idx0 * R + r] += tmp_val;
            }
        }
    }
}


int MTTKRP_TILED_HCSR_CPU(TiledTensor *TiledX, Matrix *U, const Options &Opt){
        
    // all tile same mode..doesnt matter
    ITYPE mode0 = TiledX[0].modeOrder[0];
    ITYPE mode1 = TiledX[0].modeOrder[1];
    ITYPE mode2 = TiledX[0].modeOrder[2];
    ITYPE R = Opt.R;

    for (int tile = 0; tile < Opt.nTile; ++tile){

        #pragma omp parallel
        {
            DTYPE *tmp_val = new DTYPE[R];
            DTYPE *outBuffer = new DTYPE[R];

            #pragma omp for
            for(ITYPE slc = 0; slc < TiledX[tile].sliceIdx.size(); ++slc) {
                memset(outBuffer, 0, R * sizeof(DTYPE));

                ITYPE idx0 = TiledX[tile].sliceIdx[slc];
                int fb_st = TiledX[tile].slicePtr[slc];
                int fb_end = TiledX[tile].slicePtr[slc+1];
                
                for (int fbr = fb_st; fbr < fb_end; ++fbr){
                    // ITYPE idx1 = TiledX[tile].inds[mode1][TiledX[tile].fiberPtr[fbr]];
                    
                    for(ITYPE r=0; r<R; ++r)
                        tmp_val[r] = 0;
                     
                    for(ITYPE x = TiledX[tile].fiberPtr[fbr]; x < TiledX[tile].fiberPtr[fbr+1]; ++x) {

                        ITYPE idx2 = TiledX[tile].inds[mode2][x];  
                        #pragma omp simd              
                        for(ITYPE r=0; r<R; ++r) {
                            tmp_val[r] += TiledX[tile].vals[x] * U[mode2].vals[idx2 * R + r]; 
                        }
                    }

                    ITYPE idx1 = TiledX[tile].fiberIdx[fbr];
                    // ITYPE idx1 = TiledX[tile].inds[mode1][TiledX[tile].fiberPtr[fbr]];
                    // #pragma omp simd
                    for(ITYPE r=0; r<R; ++r) 
                        outBuffer[r] += tmp_val[r] * U[mode1].vals[idx1 * R + r];//U1[idx1 * R + r];
                }
                for(ITYPE r=0; r<R; ++r) 
                    U[mode0].vals[idx0 * R + r] += outBuffer[r]; //U0[idx0 * R + r]               
            }
        }
    }
} 

// int MTTKRP_HCSR_CPU_ASR(const Tensor &X, Matrix *U, ITYPE mode, ITYPE R){
        
//     //std::fill(U[0].vals.begin(), U[0].vals.end(), 0);
//     // memset(U[0].vals, 0, U[0].nRows * U[0].nCols * sizeof(DTYPE));

//     #pragma omp parallel 
//     {
//         Matrix & restrict U0 = U[0]; mode 0
//         Matrix & restrict U1 = U[1];
//         Matrix & restrict U2 = U[2];
//         const vector<ITYPE> & restrict Xinds1 = X.inds[1];
//         const vector<ITYPE> & restrict Xinds2 = X.inds[2];
//         const vector<ITYPE> & restrict Xinds3 = X.inds[3];
//         DTYPE tmp_val[R] ;

//         #pragma omp for
//         for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {

//             // DTYPE *tmp_val = new DTYPE[R];
//             ITYPE idx0 = X.sliceIdx[slc];
//             int fb_st = X.slicePtr[slc];
//             int fb_end = X.slicePtr[slc+1];
            
//             for (int fbr = fb_st; fbr < fb_end; ++fbr){
//                 ITYPE idx1 = Xinds1[X.fiberPtr[fbr]];
//                 // ITYPE idx1 = X.inds[1][X.fiberPtr[fbr]];
                
//                 #pragma omp simd
//                 for(ITYPE r=0; r<R; ++r)
//                     tmp_val[r] = 0;
                 
//                 for(ITYPE x = X.fiberPtr[fbr]; x < X.fiberPtr[fbr+1]; ++x) 
//                 {
//                     ITYPE idx2 = Xinds2[x];
//                     auto * restrict u2_v  = &(U2.vals[idx2 * R]);
//                     auto * restrict xvals = &(X.vals[0]);

//                     #pragma omp simd
//                     for(ITYPE r = 0; r < R; ++r)
//                     {
//                         tmp_val[r] += xvals[x] * u2_v[ r];
//                     }

//                     // ITYPE idx2 = X.inds[2][x];                
//                     // for(ITYPE r=0; r<R; ++r) {
//                     //     tmp_val[r] += X.vals[x] * U[2].vals[idx2 * R + r]; 
//                     // }
//                 }
//                 auto * restrict u0_v  = &(U0.vals[idx0 * R]);
//                 auto * restrict u1_v = &(U1.vals[idx1 * R]);

//                 #pragma omp simd
//                 for(ITYPE r=0; r<R; ++r) 
//                     u0_v [r] += tmp_val[r] *  u1_v[r];
//                     // U[0].vals[idx0 * R + r] += tmp_val[r] * U[1].vals[idx1 * R + r];
                
//             }
//         }
//     }
//     return 0X00;
// }
