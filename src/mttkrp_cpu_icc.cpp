/**
 *
 * OHIO STATE UNIVERSITY SOFTWARE DISTRIBUTION LICENSE
 *
 * Load-balanced sparse MTTKRP on GPUs (the “Software”) Copyright (c) 2019, The Ohio State
 * University. All rights reserved.
 *
 * The Software is available for download and use subject to the terms and
 * conditions of this License. Access or use of the Software constitutes acceptance
 * and agreement to the terms and conditions of this License. Redistribution and
 * use of the Software in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the capitalized paragraph below.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the capitalized paragraph below in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. The names of Ohio State University, or its faculty, staff or students may not
 * be used to endorse or promote products derived from the Software without
 * specific prior written permission.
 *
 * THIS SOFTWARE HAS BEEN APPROVED FOR PUBLIC RELEASE, UNLIMITED DISTRIBUTION. THE
 * SOFTWARE IS PROVIDED “AS IS” AND WITHOUT ANY EXPRESS, IMPLIED OR STATUTORY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, WARRANTIES OF ACCURACY, COMPLETENESS,
 * NONINFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  ACCESS OR USE OF THE SOFTWARE IS ENTIRELY AT THE USER’S RISK.  IN
 * NO EVENT SHALL OHIO STATE UNIVERSITY OR ITS FACULTY, STAFF OR STUDENTS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  THE SOFTWARE
 * USER SHALL INDEMNIFY, DEFEND AND HOLD HARMLESS OHIO STATE UNIVERSITY AND ITS
 * FACULTY, STAFF AND STUDENTS FROM ANY AND ALL CLAIMS, ACTIONS, DAMAGES, LOSSES,
 * LIABILITIES, COSTS AND EXPENSES, INCLUDING ATTORNEYS’ FEES AND COURT COSTS,
 * DIRECTLY OR INDIRECTLY ARISING OUT OF OR IN CONNECTION WITH ACCESS OR USE OF THE
 * SOFTWARE.
 *
 */

/**
 *
 * Author:
 *          Israt Nisa (nisa.1@osu.edu)
 *
 * Contacts:
 *          Israt Nisa (nisa.1@osu.edu)
 *          Jiajia Li (jiajia.li@pnnl.gov)
 *
 */

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

int MTTKRP_HCSR_CPU(const Tensor &X, Matrix *U, const Options &Opt){
        
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE R = Opt.R;

    // #pragma omp parallel
    {
        DTYPE *tmp_val = new DTYPE[R];
        DTYPE *outBuffer = new DTYPE[R];

        // #pragma omp for
        for(int slc = 0; slc < X.sliceIdx.size(); ++slc) {

            memset(outBuffer, 0, R * sizeof(DTYPE));
            ITYPE idx0 = X.sliceIdx[slc];
            int fb_st = X.slicePtr[slc];
            int fb_end = X.slicePtr[slc+1];
            
            for (int fbr = fb_st; fbr < fb_end; ++fbr){
                
                for(int r=0; r<R; ++r)
                    tmp_val[r] = 0;
                 
                for(unsigned int x = X.fiberPtr[fbr]; x < X.fiberPtr[fbr+1]; ++x) {

                    ITYPE idx2 = X.inds[mode2][x];                
                    for(unsigned int r=0; r<R; ++r) {
                        tmp_val[r] += X.vals[x] * U[mode2].vals[idx2 * R + r]; 
                    }
                }
                
                ITYPE idx1 = X.fiberIdx[fbr];//X.inds[mode1][X.fiberPtr[fbr]];
                for(unsigned int r=0; r<R; ++r) 
                    outBuffer[r] += tmp_val[r] * U[mode1].vals[idx1 * R + r];
                    // U[mode0].vals[idx0 * R + r] += tmp_val[r] * U[mode1].vals[idx1 * R + r];
                
            }
            for(ITYPE r=0; r<R; ++r) 
                U[mode0].vals[idx0 * R + r] += outBuffer[r];
               // U0[idx0 * R + r] += outBuffer[r];
        }
    }
}


int MTTKRP_HCSR_CPU_ASR(const Tensor &X, Matrix *U, const Options &Opt){
        // cout << "CHANGE mode, unsigned, const " << endl;
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE R = Opt.R;
    
    #pragma omp parallel 
    {
        Matrix & restrict U0 = U[mode0];
        Matrix & restrict U1 = U[mode1];
        Matrix & restrict U2 = U[mode2];
        const vector<ITYPE> & restrict Xinds2 = X.inds[mode2];
        auto * restrict slcPtr = &(X.slicePtr[0]);
        auto * restrict slcIdx = &(X.sliceIdx[0]);
        auto * restrict fbrPtr = &(X.fiberPtr[0]);
        auto * restrict fbrIdx = &(X.fiberIdx[0]);
        auto * restrict xvals = &(X.vals[0]);
        
        DTYPE tmp_val[R] ;
        DTYPE outBuffer[R];// = new DTYPE[R];
        memset(outBuffer, 0, R * sizeof(DTYPE));
       
        #pragma omp for
        for(unsigned int slc = 0; slc < X.sliceIdx.size(); ++slc) {

            /* foreach fiber in slice */
            for (unsigned int fbr = slcPtr[slc]; fbr < slcPtr[slc+1]; ++fbr){

                for(unsigned int r=0; r<R; ++r){
                    tmp_val[r] = 0;
                }
                
                /* foreach nnz in fiber */ 
                
                for(unsigned int x = fbrPtr[fbr]; x < fbrPtr[fbr+1]; ++x)   {
                    
                    ITYPE const idx2 = Xinds2[x];
                    auto const * const restrict u2_v  = &(U2.vals[idx2 * R]);
            
                    for(unsigned int r = 0; r < R; ++r) {
                        tmp_val[r] += xvals[x] * u2_v[ r];
                    }
                }

                ITYPE const idx1 =  fbrIdx[fbr];//X.fiberIdx[fbr];//Xinds1[X.fiberPtr[fbr]];
                auto const * const restrict u1_v = &(U1.vals[idx1 * R]);
                
                for(unsigned int r=0; r<R; ++r) 
                    outBuffer[r] += tmp_val[r] *  u1_v[r];          
            }
            ITYPE const idx0 = slcIdx[slc];
            auto * restrict u0_v  = &(U0.vals[idx0 * R]);
            for(ITYPE r=0; r<R; ++r) {
                u0_v [r] += outBuffer[r] ;//*  u1_v[r];
                outBuffer[r] = 0;
            }
        }
    }
    return 0X00;
}

int MTTKRP_HYB_HCSR_CPU(HYBTensor &X, Matrix *U, Options &Opt){
        // cout << "CHANGE mode, unsigned, const " << endl;
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE R = Opt.R;
    
    #pragma omp parallel 
    {
        Matrix & restrict U0 = U[mode0];
        Matrix & restrict U1 = U[mode1];
        Matrix & restrict U2 = U[mode2];
        const vector<ITYPE> & restrict Xinds2 = X.inds[mode2];
        auto * restrict slcPtr = &(X.slicePtr[0]);
        auto * restrict slcIdx = &(X.sliceIdx[0]);
        auto * restrict fbrPtr = &(X.fiberPtr[0]);
        auto * restrict fbrIdx = &(X.fiberIdx[0]);
        auto * restrict xvals = &(X.vals[0]);
        
        DTYPE tmp_val[R] ;
        DTYPE outBuffer[R];// = new DTYPE[R];
        memset(outBuffer, 0, R * sizeof(DTYPE));
       
        #pragma omp for
        for(unsigned int slc = 0; slc < X.sliceIdx.size(); ++slc) {

            /* foreach fiber in slice */
            for (unsigned int fbr = slcPtr[slc]; fbr < slcPtr[slc+1]; ++fbr){

                for(unsigned int r=0; r<R; ++r){
                    tmp_val[r] = 0;
                }
                
                /* foreach nnz in fiber */ 
                
                for(unsigned int x = fbrPtr[fbr]; x < fbrPtr[fbr+1]; ++x)   {
                    
                    ITYPE const idx2 = Xinds2[x];
                    auto const * const restrict u2_v  = &(U2.vals[idx2 * R]);
            
                    for(unsigned int r = 0; r < R; ++r) {
                        tmp_val[r] += xvals[x] * u2_v[ r];
                    }
                }

                ITYPE const idx1 =  fbrIdx[fbr];//X.fiberIdx[fbr];//Xinds1[X.fiberPtr[fbr]];
                auto const * const restrict u1_v = &(U1.vals[idx1 * R]);
                
                for(unsigned int r=0; r<R; ++r) 
                    outBuffer[r] += tmp_val[r] *  u1_v[r];          
            }
            ITYPE const idx0 = slcIdx[slc];
            auto * restrict u0_v  = &(U0.vals[idx0 * R]);
            for(ITYPE r=0; r<R; ++r) {
                u0_v [r] += outBuffer[r] ;//*  u1_v[r];
                outBuffer[r] = 0;
            }
        }
    }
    return 0X00;
}

int MTTKRP_HYB_CSL_CPU( HYBTensor &HybX, Matrix *U, Options &Opt){
    ITYPE R = Opt.R;
    // #pragma omp parallel for //reduction(+:U[0].vals[:R])
    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];

    #pragma omp parallel 
    {
        Matrix & restrict U0 = U[mode0];
        Matrix & restrict U1 = U[mode1];
        Matrix & restrict U2 = U[mode2];
        const vector<ITYPE> & restrict Xinds1 = HybX.CSLinds[mode1];
        const vector<ITYPE> & restrict Xinds2 = HybX.CSLinds[mode2];
        auto * restrict slcPtr = &(HybX.CSLslicePtr[0]);
        auto * restrict slcIdx = &(HybX.CSLsliceIdx[0]);
        auto * restrict xvals = &(HybX.CSLvals[0]);
        
        DTYPE tmp_val[R] ;
        DTYPE outBuffer[R];// = new DTYPE[R];
        memset(outBuffer, 0, R * sizeof(DTYPE));
       
        #pragma omp for
    
        for(ITYPE slc = 0; slc < HybX.CSLsliceIdx.size(); ++slc) {

            const int fb_st = slcPtr[slc];
            const int fb_end = slcPtr[slc+1];
            
            for (int fbr = fb_st; fbr < fb_end; ++fbr){

                DTYPE tmp_val = 0;
                const ITYPE idx1 = Xinds1[fbr];
                const ITYPE idx2 = Xinds2[fbr];

                auto const * const restrict u1_v = &(U1.vals[idx1 * R]);
                auto const * const restrict u2_v = &(U2.vals[idx2 * R]);  
               
                // #pragma omp atomic
                for(ITYPE r=0; r<R; ++r) {            
                    tmp_val = xvals[fbr] * u1_v[r] * u2_v[r]; 
                    outBuffer[r] += tmp_val;
                }
            }
            ITYPE const idx0 = slcIdx[slc];
            auto * restrict u0_v  = &(U0.vals[idx0 * R]);
            for(ITYPE r=0; r<R; ++r) {
                u0_v [r] += outBuffer[r] ;//*  u1_v[r];
                outBuffer[r] = 0;
            }
        }
    }
}

int MTTKRP_HYB_CSL_CPU_naive( HYBTensor &HybX, Matrix *U, Options &Opt){
    ITYPE R = Opt.R;
    // #pragma omp parallel for //reduction(+:U[0].vals[:R])
    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    
    #pragma omp parallel 
    {
        #pragma omp for
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
}
int MTTKRP_HYB_COO_CPU_naive(const HYBTensor &HybX, Matrix *U, const Options &Opt){

    ITYPE R = Opt.R;
    // #pragma omp parallel for //reduction(+:U[0].vals[:R])
    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    
    // COO PART
    #pragma omp parallel 
    {
        #pragma omp for
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
    }
}

int MTTKRP_HYB_COO_CPU(const HYBTensor &HybX, Matrix *U, const Options &Opt){

    ITYPE R = Opt.R;
    // #pragma omp parallel for //reduction(+:U[0].vals[:R])
    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    
    // COO PART
    #pragma omp parallel 
    {
        Matrix & restrict U0 = U[mode0];
        Matrix & restrict U1 = U[mode1];
        Matrix & restrict U2 = U[mode2];
        
        const vector<ITYPE> & restrict Xinds0 = HybX.COOinds[mode0];
        const vector<ITYPE> & restrict Xinds1 = HybX.COOinds[mode1];
        const vector<ITYPE> & restrict Xinds2 = HybX.COOinds[mode2];
        auto * restrict xvals = &(HybX.COOvals[0]);
        
        #pragma omp for
        for(ITYPE x = 0; x < HybX.COOnnz; ++x) {

            DTYPE tmp_val = 0;
            ITYPE const idx0 = Xinds0[x];
            ITYPE const idx1 = Xinds1[x];
            ITYPE const idx2 = Xinds2[x];
            auto * restrict u0_v  = &(U0.vals[idx0 * R]);
            auto const * const restrict u1_v = &(U1.vals[idx1 * R]);
            auto const * const restrict u2_v = &(U2.vals[idx2 * R]);
           
            for(int r=0; r<R; ++r) {            
                u0_v [r] +=  xvals[x] *  u1_v[r] * u2_v[r]; 
            }
        }
    }
}

int MTTKRP_HYB_CPU(HYBTensor &HybX, Matrix *U, Options &Opt, int iter){
    
    // COO PART
    double t0 = seconds();    
    MTTKRP_HYB_COO_CPU(HybX, U, Opt);
    if(Opt.verbose && iter == Opt.iter - 1)
        printf("HYB COO CPU - time: %.3f sec \n", seconds() - t0);  
    
    // CSL part
    t0 = seconds(); 
    MTTKRP_HYB_CSL_CPU(HybX, U, Opt);
    if(Opt.verbose && iter == Opt.iter - 1)
        printf("HYB CSL CPU - time: %.3f sec \n", seconds() - t0);  
    
    // HCSR part
    t0 = seconds(); 
    MTTKRP_HYB_HCSR_CPU(HybX, U, Opt);
    if(Opt.verbose &&  iter == Opt.iter - 1)
        printf("HYB HCSR CPU - time: %.3f sec \n", seconds() - t0);  
    

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
