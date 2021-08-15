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

#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <utility>  
#include <math.h> 
#include <omp.h>
#include <cuda.h>
#include "mttkrp_cpu.h"
#include <bits/stdc++.h>  
// #include <papi.h>
#include "mttkrp_gpu.h" 
using namespace std;

void handle_error(int err){
    printf("PAPI error %d \n", err); 
}

int main(int argc, char* argv[]){ 

    // int numEvents = 7;
    // long long *values = new long long [numEvents];

    // int events[7] = {PAPI_L1_DCM, PAPI_L2_DCM,
    //     PAPI_RES_STL,  PAPI_L3_TCM,
    //     PAPI_LD_INS, PAPI_SR_INS, PAPI_BR_INS
    //     };

    // int events[1] = {PAPI_L1_DCM, PAPI_L2_DCM, PAPI_L3_DCA,
    // PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM,
    // PAPI_RES_STL, PAPI_LST_INS,
    // PAPI_PRF_DM, PAPI_LD_INS, PAPI_SP_OPS, PAPI_VEC_SP};

  //   int retval = PAPI_library_init(PAPI_VER_CURRENT);
  //   if (retval != PAPI_VER_CURRENT) {
  //       printf("Error! PAPI_library_init %d\n",retval);
  // }
 
    Options Opt = parse_cmd_options(argc, argv);
    // Opt.print();
    Tensor X;
    load_tensor(X, Opt.inFileName);

    //TBD:: sort X
    // tensor_stats(X);
    // print_COOtensor(X);
    
    //TBD:: fix hard coded 3
    Matrix U[3];   
    for (int i = 0; i < X.ndims; ++i){
       create_mats(X, U, i, Opt.R);
       randomize_mats(X, U, i, Opt.R);
    }
    zero_mat(X, U, Opt.mode);

    if(Opt.verbose)
        cout << endl << "Starting MTTKRP..." << endl;  
    
    // COO CPU   
    if(Opt.impType == 1){
        double t0 = seconds();
        MTTKRP_COO_CPU(X, U, Opt.mode, Opt.R);
        printf("COO CPU - time: %.3f sec \n", seconds() - t0);
    }

    // HCSR CPU   
    else if(Opt.impType == 2){
        create_HCSR(X);
        // print_HCSRtensor(X);
        double t0 = seconds();
        // if (PAPI_start_counters(events, numEvents) != PAPI_OK)
        //     handle_error(1);
        MTTKRP_HCSR_CPU(X, U, Opt.mode, Opt.R);
        // if ( PAPI_stop_counters(values, numEvents) != PAPI_OK)
        //     handle_error(1);

      // for (int nm = 0; nm < numEvents; ++nm)
      //     printf(" %d " , values[nm]);
      // printf("\n");
        printf("HCSR CPU - time: %.3f sec \n", seconds() - t0);
    }

    // COO GPU  
    else if(Opt.impType == 3){
        cout << " GPU COO has bugs! " << endl;
        MTTKRP_COO_GPU(X, U, Opt);
    }

    // HCSR GPU  
    else if(Opt.impType == 4){
        create_HCSR(X);
        MTTKRP_HCSR_GPU(X, U, Opt);
    }

    // Tiled versions
    else if(Opt.impType >= 5){
        
        TiledTensor TiledX[Opt.nTile];
        
        create_HCSR(X);
        
        Opt.tileSize = (X.dims[2] + Opt.nTile - 1)/Opt.nTile;
        if(Opt.nTile > X.dims[2]){
            cout << "Number of tiles ("<< Opt.nTile << ") should be as minimum as K's dimension (" << X.dims[2]  << "). Exiting."<< endl ;
            exit(0);
        }

        // split X into tiles based on K indices
        make_KTiling(X, TiledX, Opt);
        
        // create HCSR for each tile
        for (int tile = 0; tile < Opt.nTile; ++tile){
            create_TiledHCSR(TiledX, tile);
            // print_TiledHCSRtensor(TiledX, tile);
        }

        // Split tiles into bins accordin to nnz in slice
        for (int tile = 0; tile < Opt.nTile; ++tile){
            make_TiledBin(TiledX, Opt, tile);
        }

        // MTTKRP 
        // COO GPU  
        if(Opt.impType == 5){
            double t0 = seconds();
            MTTKRP_TILED_COO_CPU(TiledX, U, Opt); 
            printf("TILED COO CPU - time: %.3f sec \n", seconds() - t0);  
        }

         // HCSR GPU  
        else if(Opt.impType == 6){
            double t0 = seconds();
            MTTKRP_TILED_HCSR_CPU(TiledX, U, Opt); 
            printf("TILED HCSR CPU - time: %.3f sec \n", seconds() - t0); 
        }  

        //COO GPU 
        else if(Opt.impType == 7){
            cout << "GPU COO has bugs! " << endl;
            MTTKRP_TILED_COO_GPU(TiledX, U, Opt);
        }

        // HCSR GPU
        else if(Opt.impType == 8){
            MTTKRP_TILED_HCSR_GPU(TiledX, U, Opt);
        }
    }
    else // e.g. -1 
        cout << "no MTTKRP" << endl;

    if(!Opt.outFileName.empty())
        write_output(U, Opt.mode, Opt.outFileName);
}


