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

    int *curMode = new int [X.ndims];
    ITYPE R = Opt.R;

    for (int m = 0; m < X.ndims; ++m)
        curMode[m] = (m + Opt.mode) % X.ndims;
           
    ITYPE mode0 = curMode[0];
    ITYPE mode1 = curMode[1];
    ITYPE mode2 = curMode[2];
    
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

int MTTKRP_COO_CPU_4D(const Tensor &X, Matrix *U, const Options &Opt){
    
    int *curMode = new int [X.ndims];
    ITYPE R = Opt.R;
    for (int m = 0; m < X.ndims; ++m)
        curMode[m] = (m + Opt.mode) % X.ndims;

    ITYPE mode0 = curMode[0];
    ITYPE mode1 = curMode[1];
    ITYPE mode2 = curMode[2];
    ITYPE mode3 = curMode[3];

    for(ITYPE x=0; x<X.totNnz; ++x) {

        DTYPE tmp_val = 0;
        ITYPE idx0 = X.inds[mode0][x];
        ITYPE idx1 = X.inds[mode1][x];
        ITYPE idx2 = X.inds[mode2][x];
        ITYPE idx3 = X.inds[mode3][x];
       
        // #pragma omp atomic
        for(ITYPE r=0; r<R; ++r) {            
            tmp_val = X.vals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r] * U[mode3].vals[idx3 * R + r];
            U[mode0].vals[idx0 * R + r] += tmp_val;
        }
    }
}


int MTTKRP_HCSR_CPU_slc(const Tensor &X, const TiledTensor *TiledX, Matrix *U, const Options &Opt){
        
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE R = Opt.R;

     cout << "mode 0: " << mode0 << " " << mode1 <<" " << mode2 << endl;

     // MTTKRP_HCSR_CPU_RSTRCT(const Tensor &X, U[], const Options &Opt)
    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;
 
    DTYPE *tmp_val = new DTYPE[R];
    DTYPE *outBuffer = new DTYPE[R];

    // #pragma omp for
    for (int fbr = 0; fbr <  X.nFibers; ++fbr){

        memset(tmp_val, 0, R * sizeof(DTYPE));
        
        ITYPE idx1 = X.fbrIdx[1][fbr];
        ITYPE idx0 = X.fbrLikeSlcInds[fbr]; //2S +2F + M  now 3F + M
       
        for(ITYPE r=0; r<R; ++r){
            tmp_val[r] = 0;
        }
         
        for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {

            ITYPE idx2 = X.inds[mode2][x];  

            for(ITYPE r=0; r<R; ++r) 
                tmp_val[r] += X.vals[x] * U2[idx2 * R + r]; 
        }

        for(ITYPE r=0; r<R; ++r) 
             U0[idx0 * R + r] += tmp_val[r] * U1[idx1 * R + r];           
    }
}

int MTTKRP_HCSR_CPU(const Tensor &X, const TiledTensor *TiledX, Matrix *U, const Options &Opt){

    // if(Opt.impType == 11)
    //     TiledTensor X = TiledX[Opt.mode];
    // else 
    //     Tensor X = XX;
        
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE R = Opt.R;

     cout << "mode 0: " << mode0 << " " << mode1 <<" " << mode2 << endl;

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
        for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
            memset(outBuffer, 0, R * sizeof(DTYPE));

            ITYPE idx0 = X.fbrIdx[0][slc];
            
            for (int fbr = X.fbrPtr[0][slc]; fbr <  X.fbrPtr[0][slc+1]; ++fbr){

                for(ITYPE r=0; r<R; ++r){
                    tmp_val[r] = 0;
                }
                 
                for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {

                    ITYPE idx2 = X.inds[mode2][x];  
        
                    for(ITYPE r=0; r<R; ++r) {
                        tmp_val[r] += X.vals[x] * U2[idx2 * R + r]; 
                    }
                }
                
                ITYPE idx1 = X.fbrIdx[1][fbr];

                for(ITYPE r=0; r<R; ++r) 
                    outBuffer[r] += tmp_val[r] * U1[idx1 * R + r];               
            }
            for(ITYPE r=0; r<R; ++r) 
                U0[idx0 * R + r] += outBuffer[r];
        }
    }
}

int MTTKRP_HCSR_CPU_mode1(const Tensor &X, Matrix *U, const Options &Opt, const int mode){
        
    // int *curMode = new int [X.ndims];
    
    // for (int m = 0; m < X.ndims; ++m)
    //     curMode[m] = (X.modeOrder[m] + mode) % X.ndims;
           
    // ITYPE mode0 = curMode[0]; //1
    // ITYPE mode1 = curMode[1]; //0
    // ITYPE mode2 = curMode[2]; //2


    ITYPE mode0 = mode; // TiledX[MTTKRPmode].modeOrder[0]; // 0
    ITYPE mode1 = X.modeOrder[2] ;//TiledX[HCSRmode].modeOrder[2]; // 1
    ITYPE mode2 = X.modeOrder[0] ;//iledX[HCSRmode].modeOrder[0];// 2

    cout << "mode 1: " << mode0 << " " << mode1 <<" " << mode2 << endl;
    ITYPE R = Opt.R;

     // MTTKRP_HCSR_CPU_RSTRCT(const Tensor &X, U[], const Options &Opt)
    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;

    DTYPE *tmp_val = new DTYPE[R];
  
    // #pragma omp parallel
    {    
     memset(U0, 0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));

        // #pragma omp for
        // for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) 
        {
           
            // ITYPE idx2 = X.fbrIdx[0][slc];
            
            // for (int fbr = X.fbrPtr[0][slc]; fbr <  X.fbrPtr[0][slc+1]; ++fbr){
            for (int fbr = 0; fbr <  X.nFibers; ++fbr){

                memset(tmp_val, 0, R * sizeof(DTYPE));
                
                ITYPE idx0 = X.fbrIdx[1][fbr];
                ITYPE idx2 = X.fbrLikeSlcInds[fbr]; //2S +2F + M  now 3F + M
                 
                for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {

                    ITYPE idx1 = X.inds[X.modeOrder[2]][x];  
        
                    for(ITYPE r=0; r<R; ++r) {
                        tmp_val[r] += X.vals[x] * U2[idx2 * R + r] * U1[idx1 * R + r]; 
                    }
                } 
                for(ITYPE r=0; r<R; ++r)
                    U0[idx0 * R + r] +=  tmp_val[r];
            }
        }
    }
}

/* MTTKRP on mode 2 using sorted tensor 0-1-2  */

int MTTKRP_HCSR_CPU_mode2(const Tensor &X, Matrix *U, const Options &Opt, const int mode){

    // int *curMode = new int [X.ndims];
    
    // for (int m = 0; m < X.ndims; ++m)
    //     curMode[m] = (m + mode) % X.ndims;
           
    // ITYPE mode0 = curMode[0];
    // ITYPE mode1 = curMode[1];
    // ITYPE mode2 = curMode[2];

    // int *curMode = new int [X.ndims];
    
    // for (int m = 0; m < X.ndims; ++m)
    //     curMode[m] = (X.modeOrder[m] + mode) % X.ndims;
           
    // ITYPE mode0 = curMode[0];
    // ITYPE mode1 = curMode[1];
    // ITYPE mode2 = curMode[2];


    ITYPE mode0 = mode; // TiledX[MTTKRPmode].modeOrder[0]; // 0
    ITYPE mode1 = X.modeOrder[0] ;//TiledX[HCSRmode].modeOrder[2]; // 1
    ITYPE mode2 = X.modeOrder[1] ;//iledX[HCSRmode].modeOrder[0];// 2

    cout << "mode 2: " << mode0 << " " << mode1 <<" " << mode2 << endl;
        
    // ITYPE mode0 = 2; //X.modeOrder[0];
    // ITYPE mode1 = 0; //X.modeOrder[1];
    // ITYPE mode2 = 1; //X.modeOrder[2];
    ITYPE R = Opt.R;

     // MTTKRP_HCSR_CPU_RSTRCT(const Tensor &X, U[], const Options &Opt)
    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;

    DTYPE *tmp_val = new DTYPE[R];
    // memset(tmp_val, 0, R * sizeof(DTYPE));

    // ITYPE const * const __restrict__ arrIdx2 = &(X.inds[mode2][0]);
  
    // #pragma omp parallel
    {    
     memset(U0, 0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));

        // #pragma omp for
        for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
           
            ITYPE idx1 = X.fbrIdx[0][slc];
            
            for (int fbr = X.fbrPtr[0][slc]; fbr <  X.fbrPtr[0][slc+1]; ++fbr){
                
                ITYPE idx2 = X.fbrIdx[1][fbr];

                for(ITYPE r=0; r<R; ++r) 
                    tmp_val[r] = U2[idx2 * R + r] * U1[idx1 * R + r]; 
                    
                 
                for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {

                    ITYPE idx0 = X.inds[X.modeOrder[2]][x];  
        
                    for(ITYPE r=0; r<R; ++r) {
                        U0[idx0 * R + r] += X.vals[x] * tmp_val[r]; 
                        // U0[idx0 * R + r] += X.vals[x] * U2[idx2 * R + r] * U1[idx1 * R + r]; 
                    }
                }       
            }
        }
    }
}

int MTTKRP_MIHCSR_CPU(const TiledTensor *TiledX, Matrix *U, const Options &Opt, const int mode){

    TiledTensor X = TiledX[mode];

    int *curMode = new int [X.ndims];

    // for (int m = 0; m < X.ndims; ++m)
    //     curMode[m] = (m + mode) % X.ndims;
           
    // ITYPE mode0 = curMode[0];
    // ITYPE mode1 = curMode[1];
    // ITYPE mode2 = curMode[2];
        
    
    // ITYPE mode0 = X.modeOrder[0];
    // ITYPE mode1 = X.modeOrder[1];
    // ITYPE mode2 = X.modeOrder[2];

    ITYPE mode0 = TiledX[mode].modeOrder[0]; // 0
    ITYPE mode1 = TiledX[mode].modeOrder[1]; // 2
    ITYPE mode2 = TiledX[mode].modeOrder[2]; // 1
    ITYPE R = Opt.R;

    cout << "NO ATOMIC " << mode0 << " " << mode1 <<" " << mode2 << endl;

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
        for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
            memset(outBuffer, 0, R * sizeof(DTYPE));

            ITYPE idx0 = X.fbrIdx[0][slc];
            
            for (int fbr = X.fbrPtr[0][slc]; fbr <  X.fbrPtr[0][slc+1]; ++fbr){

                for(ITYPE r=0; r<R; ++r){
                    tmp_val[r] = 0;
                }
                 
                for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {

                    ITYPE idx2 = X.inds[mode2][x];  
        
                    for(ITYPE r=0; r<R; ++r) {
                        tmp_val[r] += X.vals[x] * U2[idx2 * R + r]; //2MR
                    }
                }
                
                ITYPE idx1 = X.fbrIdx[1][fbr];

                for(ITYPE r=0; r<R; ++r) 
                    outBuffer[r] += tmp_val[r] * U1[idx1 * R + r];    // 2PR           
            }
            for(ITYPE r=0; r<R; ++r) 
                U0[idx0 * R + r] += outBuffer[r]; //IR
        }
    }
}



/* MTTKRP on mode 0 using sorted tensor tile 2-0-1  */

int MTTKRP_MIHCSR_CPU_FBR_ATOMICS(const TiledTensor *TiledX, Matrix *U, const Options &Opt, const int HCSRmode, const int MTTKRPmode){

    TiledTensor X = TiledX[HCSRmode];

    // MTTKRP mode
    // int mode = MTTKRPmode;// 1;//0; if mode 1
    
    int *mode = new int [X.ndims];

    // for (int m = 0; m < X.ndims; ++m)
    //     curMode[m] = (m + mode) % X.ndims;
           
    // ITYPE mode0 = curMode[0];
    // ITYPE mode1 = curMode[1];
    // ITYPE mode2 = curMode[2];

    // ITYPE mode0 = TiledX[mode].modeOrder[0]; // 0
    // ITYPE mode1 = TiledX[mode].modeOrder[1]; // 1
    // ITYPE mode2 = TiledX[mode].modeOrder[2]; // 2

    ITYPE mode0 = TiledX[MTTKRPmode].modeOrder[0]; // 0
    ITYPE mode1 = TiledX[HCSRmode].modeOrder[2]; // 1
    ITYPE mode2 = TiledX[HCSRmode].modeOrder[0];// 2


    cout << "FBR ATOMIC " << mode0 << " " << mode1 <<" " << mode2 << endl;
        
    ITYPE R = Opt.R;

    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;

    DTYPE *tmp_val = new DTYPE[R];
    
    // #pragma omp for
    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
       
        ITYPE idx2 = X.fbrIdx[0][slc];
        
        for (int fbr = X.fbrPtr[0][slc]; fbr <  X.fbrPtr[0][slc+1]; ++fbr){

            memset(tmp_val, 0, R * sizeof(DTYPE));
            
            ITYPE idx0 = X.fbrIdx[1][fbr];
             
            for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {

                ITYPE idx1 = X.inds[mode1][x];  
    
                for(ITYPE r=0; r<R; ++r) {

                    tmp_val[r] += X.vals[x] * U1[idx1 * R + r]; //2MR
                }
            }

            for(ITYPE r=0; r<R; ++r)
                U0[idx0 * R + r] += tmp_val[r] * U2[idx2 * R + r]; //2PR
        }
    }
}

/* MTTKRP on mode 1 using sorted tensor tile 2-0-1  */

int MTTKRP_MIHCSR_CPU_ALL_ATOMICS(const TiledTensor *TiledX, Matrix *U, const Options &Opt, const int HCSRmode, const int MTTKRPmode){

    TiledTensor X = TiledX[HCSRmode];

    // MTTKRP mode
    int mode = MTTKRPmode;
    
    // int *curMode = new int [X.ndims];

    // for (int m = 0; m < X.ndims; ++m)
    //     curMode[m] = (m + mode) % X.ndims;
           
    // ITYPE mode0 = curMode[0]; //1
    // ITYPE mode1 = curMode[1]; //2
    // ITYPE mode2 = curMode[2]; //0

    ITYPE mode0 = TiledX[MTTKRPmode].modeOrder[0]; // 0
    ITYPE mode1 = TiledX[HCSRmode].modeOrder[0]; // 1
    ITYPE mode2 = TiledX[HCSRmode].modeOrder[1];// 2

    cout << "ALL ATOMIC " << mode0 << " " << mode1 <<" " << mode2 << endl;
    ITYPE R = Opt.R;

    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;

    DTYPE *tmp_val = new DTYPE[R];
    
    // #pragma omp for
    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
       
        ITYPE idx1 = X.fbrIdx[0][slc];
        
        for (int fbr = X.fbrPtr[0][slc]; fbr <  X.fbrPtr[0][slc+1]; ++fbr){

            memset(tmp_val, 0, R * sizeof(DTYPE));
            
            ITYPE idx2 = X.fbrIdx[1][fbr];
             
            for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {

                ITYPE idx0 = X.inds[mode0][x]; 
    
                for(ITYPE r=0; r<R; ++r) {

                    U0[idx0 * R + r] += X.vals[x] * U2[idx2 * R + r] * U1[idx1 * R + r]; 
                }
            }
        }
    }
}

int MTTKRP_HCSR_CPU_4D(const Tensor &X, Matrix *U, const Options &Opt){
        
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE mode3 = X.modeOrder[3];
    ITYPE R = Opt.R;

     // MTTKRP_HCSR_CPU_RSTRCT(const Tensor &X, U[], const Options &Opt)
    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;
    DTYPE const * const  U3 = U[mode3].vals;

    // ITYPE const * const __restrict__ arrIdx2 = &(X.inds[mode2][0]);
  
    // #pragma omp parallel
    {    
                    
        DTYPE *tmp_val = new DTYPE[R];
        DTYPE *outBuffer = new DTYPE[R];
        DTYPE *outBuffer1 = new DTYPE[R];

        // #pragma omp for
        for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
               
            memset(outBuffer, 0, R * sizeof(DTYPE));
            
            for (int fbrS = X.fbrPtr[0][slc]; fbrS <  X.fbrPtr[0][slc+1]; ++fbrS){

         
                memset(outBuffer1, 0, R * sizeof(DTYPE));
                
                for (int fbr = X.fbrPtr[1][fbrS]; fbr < X.fbrPtr[1][fbrS+1]; ++fbr){        
             
                    memset(tmp_val, 0, R * sizeof(DTYPE));
                     
                    for(ITYPE x = X.fbrPtr[2][fbr]; x < X.fbrPtr[2][fbr+1]; ++x) {
                        
                        ITYPE idx3 = X.inds[mode3][x];  
            
                        for(ITYPE r=0; r<R; ++r) 
                            tmp_val[r] += X.vals[x] * U3[idx3 * R + r]; 
                    } 
                    ITYPE idx2 = X.fbrIdx[2][fbr];              
                    for(ITYPE r=0; r<R; ++r) 
                        outBuffer1[r] += tmp_val[r] * U2[idx2 * R + r];    
                } 
                ITYPE idx1 = X.fbrIdx[1][fbrS]; 
                for(ITYPE r=0; r<R; ++r) 
                    outBuffer[r] += outBuffer1[r] * U1[idx1 * R + r];   
            }
            ITYPE idx0 = X.fbrIdx[0][slc];

            for(ITYPE r=0; r<R; ++r) 
                U0[idx0 * R + r] += outBuffer[r];
        }
    }
}

int MTTKRP_HYB_HCSR_CPU_4D(const HYBTensor &X, Matrix *U, const Options &Opt){
        
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE mode3 = X.modeOrder[3];
    ITYPE R = Opt.R;

     // MTTKRP_HCSR_CPU_RSTRCT(const Tensor &X, U[], const Options &Opt)
    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;
    DTYPE const * const  U3 = U[mode3].vals;

    // ITYPE const * const __restrict__ arrIdx2 = &(X.inds[mode2][0]);
  
    // #pragma omp parallel
    {    
                    
        DTYPE *tmp_val = new DTYPE[R];
        DTYPE *outBuffer = new DTYPE[R];
        DTYPE *outBuffer1 = new DTYPE[R];

        // #pragma omp for
        for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
               
            memset(outBuffer, 0, R * sizeof(DTYPE));
            
            for (int fbrS = X.fbrPtr[0][slc]; fbrS <  X.fbrPtr[0][slc+1]; ++fbrS){

         
                memset(outBuffer1, 0, R * sizeof(DTYPE));
                
                for (int fbr = X.fbrPtr[1][fbrS]; fbr < X.fbrPtr[1][fbrS+1]; ++fbr){        
             
                    memset(tmp_val, 0, R * sizeof(DTYPE));
                     
                    for(ITYPE x = X.fbrPtr[2][fbr]; x < X.fbrPtr[2][fbr+1]; ++x) {
                        
                        ITYPE idx3 = X.inds[mode3][x];  
            
                        for(ITYPE r=0; r<R; ++r) 
                            tmp_val[r] += X.vals[x] * U3[idx3 * R + r]; 
                    } 
                    ITYPE idx2 = X.fbrIdx[2][fbr];              
                    for(ITYPE r=0; r<R; ++r) 
                        outBuffer1[r] += tmp_val[r] * U2[idx2 * R + r];    
                } 
                ITYPE idx1 = X.fbrIdx[1][fbrS]; 
                for(ITYPE r=0; r<R; ++r) 
                    outBuffer[r] += outBuffer1[r] * U1[idx1 * R + r];   
            }
            ITYPE idx0 = X.fbrIdx[0][slc];

            for(ITYPE r=0; r<R; ++r) 
                U0[idx0 * R + r] += outBuffer[r];
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

int MTTKRP_HYB_COO_CPU_naive_4D(const HYBTensor &HybX, Matrix *U, const Options &Opt){

    ITYPE R = Opt.R;
    // #pragma omp parallel for //reduction(+:U[0].vals[:R])
    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    ITYPE mode3 = HybX.modeOrder[3];
    
    // COO PART
    #pragma omp parallel 
    {
        #pragma omp for
        for(ITYPE x = 0; x < HybX.COOnnz; ++x) {

            DTYPE tmp_val = 0;
            ITYPE idx0 = HybX.COOinds[mode0][x];
            ITYPE idx1 = HybX.COOinds[mode1][x];
            ITYPE idx2 = HybX.COOinds[mode2][x];
            ITYPE idx3 = HybX.COOinds[mode3][x];
           
            // #pragma omp atomic
            for(ITYPE r=0; r<R; ++r) {            
                tmp_val = HybX.COOvals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r] *  U[mode3].vals[idx3 * R + r];
                U[mode0].vals[idx0 * R + r] += tmp_val;
            }
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
        for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
            memset(outBuffer, 0, R * sizeof(DTYPE));

            ITYPE idx0 = X.fbrIdx[0][slc];
            const int fb_st = X.fbrPtr[0][slc];
            const int fb_end = X.fbrPtr[0][slc+1];
            
            for (int fbr = fb_st; fbr < fb_end; ++fbr){
                #pragma omp simd
                for(ITYPE r=0; r<R; ++r){
                    tmp_val[r] = 0;
                }
                 
                for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {

                    ITYPE idx2 = X.inds[mode2][x];  
                    // ITYPE idx2 = arrIdx2[x];  
                    #pragma omp simd              
                    for(ITYPE r=0; r<R; ++r) {
                        tmp_val[r] += X.vals[x] * U2[idx2 * R + r]; 
                    }
                }
                
                ITYPE idx1 = X.fbrIdx[1][fbr];
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

    // COO part
    MTTKRP_HYB_COO_CPU_naive(HybX, U, Opt);
    // CSSL part
    MTTKRP_HYB_CSL_CPU(HybX, U, Opt);
    // HCSR part
    MTTKRP_HYB_HCSR_CPU(HybX, U, Opt);

}

int MTTKRP_HYB_CPU_4D( HYBTensor &HybX, Matrix *U, Options &Opt){
    cout << "HYB CPU " << endl;
    // COO part
    MTTKRP_HYB_COO_CPU_naive_4D(HybX, U, Opt);
    // not using CSSL part for 4D
    // MTTKRP_HYB_CSL_CPU(HybX, U, Opt);

    /* HCSR part */
    MTTKRP_HYB_HCSR_CPU_4D(HybX, U, Opt);

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
            for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {
                memset(outBuffer, 0, R * sizeof(DTYPE));

                ITYPE idx0 = TiledX[tile].fbrIdx[0][slc];
                int fb_st = TiledX[tile].fbrPtr[0][slc];
                int fb_end = TiledX[tile].fbrPtr[0][slc+1];
                
                for (int fbr = fb_st; fbr < fb_end; ++fbr){
                    // ITYPE idx1 = TiledX[tile].inds[mode1][TiledX[tile].fbrPtr[1][fbr]];
                    
                    for(ITYPE r=0; r<R; ++r)
                        tmp_val[r] = 0;
                     
                    for(ITYPE x = TiledX[tile].fbrPtr[1][fbr]; x < TiledX[tile].fbrPtr[1][fbr+1]; ++x) {

                        ITYPE idx2 = TiledX[tile].inds[mode2][x];  
                        #pragma omp simd              
                        for(ITYPE r=0; r<R; ++r) {
                            tmp_val[r] += TiledX[tile].vals[x] * U[mode2].vals[idx2 * R + r]; 
                        }
                    }

                    ITYPE idx1 = TiledX[tile].fbrIdx[1][fbr];
                    // ITYPE idx1 = TiledX[tile].inds[mode1][TiledX[tile].fbrPtr[1][fbr]];
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



int MTTKRP_TILED_HCSR_CPU_4D(TiledTensor *TiledX, Matrix *U, const Options &Opt){
        
    ITYPE mode0 = TiledX[0].modeOrder[0];
    ITYPE mode1 = TiledX[0].modeOrder[1];
    ITYPE mode2 = TiledX[0].modeOrder[2];
    ITYPE mode3 = TiledX[0].modeOrder[3];
    ITYPE R = Opt.R;

     // MTTKRP_HCSR_CPU_RSTRCT(const Tensor &X, U[], const Options &Opt)
    DTYPE * U0 = U[mode0].vals;
    DTYPE const * const  U1 = U[mode1].vals;
    DTYPE const * const  U2 = U[mode2].vals;
    DTYPE const * const  U3 = U[mode3].vals;

    // ITYPE const * const __restrict__ arrIdx2 = &(TiledX[tile].inds[mode2][0]);
    for (int tile = 0; tile < Opt.nTile; ++tile){  
        // #pragma omp parallel
        {    
                        
            DTYPE *tmp_val = new DTYPE[R];
            DTYPE *outBuffer = new DTYPE[R];
            DTYPE *outBuffer1 = new DTYPE[R];

            // #pragma omp for
            for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {
                
         
                memset(outBuffer, 0, R * sizeof(DTYPE));
                
                for (int fbrS = TiledX[tile].fbrPtr[0][slc]; fbrS <  TiledX[tile].fbrPtr[0][slc+1]; ++fbrS){

             
                    memset(outBuffer1, 0, R * sizeof(DTYPE));
                    
                    for (int fbr = TiledX[tile].fbrPtr[1][fbrS]; fbr < TiledX[tile].fbrPtr[1][fbrS+1]; ++fbr){        
                 
                        memset(tmp_val, 0, R * sizeof(DTYPE));
                         
                        for(ITYPE x = TiledX[tile].fbrPtr[2][fbr]; x < TiledX[tile].fbrPtr[2][fbr+1]; ++x) {
                            
                            ITYPE idx3 = TiledX[tile].inds[mode3][x];  
                
                            for(ITYPE r=0; r<R; ++r) 
                                tmp_val[r] += TiledX[tile].vals[x] * U3[idx3 * R + r]; 
                        } 
                        ITYPE idx2 = TiledX[tile].fbrIdx[2][fbr];              
                        for(ITYPE r=0; r<R; ++r) 
                            outBuffer1[r] += tmp_val[r] * U2[idx2 * R + r];    
                    } 
                    ITYPE idx1 = TiledX[tile].fbrIdx[1][fbrS]; 
                    for(ITYPE r=0; r<R; ++r) 
                        outBuffer[r] += outBuffer1[r] * U1[idx1 * R + r];   
                }
                ITYPE idx0 = TiledX[tile].fbrIdx[0][slc];
                for(ITYPE r=0; r<R; ++r) 
                    U0[idx0 * R + r] += outBuffer[r];
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
//         for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

//             // DTYPE *tmp_val = new DTYPE[R];
//             ITYPE idx0 = X.fbrIdx[0][slc];
//             int fb_st = X.fbrPtr[0][slc];
//             int fb_end = X.fbrPtr[0][slc+1];
            
//             for (int fbr = fb_st; fbr < fb_end; ++fbr){
//                 ITYPE idx1 = Xinds1[X.fbrPtr[1][fbr]];
//                 // ITYPE idx1 = X.inds[1][X.fbrPtr[1][fbr]];
                
//                 #pragma omp simd
//                 for(ITYPE r=0; r<R; ++r)
//                     tmp_val[r] = 0;
                 
//                 for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) 
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
