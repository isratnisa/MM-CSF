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