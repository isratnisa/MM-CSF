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
#include "mttkrp_gpu.h" 
#include "cpd_cpu.h"
#include <bits/stdc++.h> 
 
using namespace std;

int main(int argc, char* argv[]){ 
 
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    Options Opt = parse_cmd_options(argc, argv);
    
    Tensor X;
    load_tensor(X, Opt);
    double t1=seconds();;
    sort_COOtensor(X);
    // printf("Sort : %.3f sec \n", seconds() - t1); 
    check_opt(X, Opt); //check options are good
    
    TiledTensor TiledX[Opt.nTile];
      
    Matrix *U = new Matrix[X.ndims]; 
    create_mats(X, U, Opt, false);
    randomize_mats(X, U, Opt);
    // if(Opt.impType != 12 && Opt.impType != 14 )
    //     zero_mat(X, U, Opt.mode); // not sure about the cpu code 

    if(Opt.verbose)
        cout << endl << "Starting MTTKRP..." << endl;  
    
    // print tensors and statistics
    if(Opt.impType == 0){
        double t0 = seconds();
        // print_COOtensor(X);
        create_HCSR(X, Opt);
        get_nnzPerFiberData(X);
        tensor_stats(X);

        // ((X.ndims == 3) ? print_HCSRtensor(X) : print_HCSRtensor_4D(X));  
    }
    // COO CPU   
    if(Opt.impType == 1){
        double t0 = seconds();
        ((X.ndims == 3) ?  MTTKRP_COO_CPU(X, U, Opt) :  MTTKRP_COO_CPU_4D(X, U, Opt));   
        printf("COO CPU - time: %.3f sec \n", seconds() - t0);
    }

    // HCSR CPU   
    else if(Opt.impType == 2){
        
        create_HCSR(X, Opt); 

        int *curMode = new int [X.ndims];
    
        for (int m = 0; m < X.ndims; ++m)
            curMode[m] = (m + Opt.mode) % X.ndims; 

        double t0 = seconds();        
        ((X.ndims == 3) ?  MTTKRP_HCSR_CPU(X, TiledX, U, Opt) :  MTTKRP_HCSR_CPU_4D(X, U, Opt)); 
        printf("gcc no opt : HCSR CPU - time: %.3f sec \n", seconds() - t0);        
    }

    // COO GPU  
    else if(Opt.impType == 3){
        MTTKRP_COO_GPU(X, U, Opt);
    }

    // HCSR GPU  
    else if(Opt.impType == 4){
        
        create_HCSR(X, Opt);
        MTTKRP_HCSR_GPU(X, U, Opt);
    }
    // HYB CPU
    else if(Opt.impType == 10){
        
        create_HCSR(X, Opt);
        HYBTensor HybX(X);
        cout << "Creating HYB... " ;
        double t0 = seconds();
        ((X.ndims == 3) ?  create_HYB(HybX, X, Opt) :  create_HYB_4D(HybX, X, Opt));   
        printf("create HYB - time: %.3f sec \n", seconds() - t0);
        make_HybBin(HybX, Opt);
        // print_HYBtensor(HybX);      
        // ((X.ndims == 3) ?  MTTKRP_HYB_CPU(HybX, U, Opt) :  MTTKRP_HYB_CPU_4D(HybX, U, Opt));   
        MTTKRP_HYB_GPU(HybX, U, Opt);      
    }

    /* Tiled versions */
    else if(Opt.impType >= 5 && Opt.impType < 10){

        create_HCSR(X, Opt);

        int tilingMode = X.modeOrder[X.ndims -1];

        Opt.tileSize = (X.dims[tilingMode] + Opt.nTile - 1)/Opt.nTile;  
        
        if(Opt.nTile > X.dims[tilingMode]){
            cout << "Number of tiles ("<< Opt.nTile << ") should be as minimum as K's dimension (" << X.dims[tilingMode]  << "). Exiting."<< endl ;
            exit(0);
        }

        // split X into tiles based on K indices
        make_KTiling(X, TiledX, Opt);
        
        // create HCSR for each tile
        for (int tile = 0; tile < Opt.nTile; ++tile){
            double t0 = seconds();
            if(TiledX[tile].totNnz > 0)
                create_TiledHCSR(TiledX, Opt, tile);
            printf("CREATE CSF : %.3f sec \n", seconds() - t0); 
            // print_TiledHCSRtensor(TiledX, tile);
        }  

        // Split tiles into bins accordin to nnz in slice
        for (int tile = 0; tile < Opt.nTile; ++tile){
             double t0 = seconds();
            if(TiledX[tile].totNnz > 0)
                make_TiledBin(TiledX, Opt, tile);
            printf("Binning time : %.3f sec \n", seconds() - t0); 
        }

        // COO GPU  
        if(Opt.impType == 5){
            double t0 = seconds();
            MTTKRP_TILED_COO_CPU(TiledX, U, Opt); 
            printf("TILED COO CPU - time: %.3f sec \n", seconds() - t0);  
        }

         // HCSR GPU  
        else if(Opt.impType == 6){
            double t0 = seconds();
            ((X.ndims == 3) ? MTTKRP_TILED_HCSR_CPU(TiledX, U, Opt) : MTTKRP_TILED_HCSR_CPU_4D(TiledX, U, Opt)); 
            printf("TILED HCSR CPU - time: %.3f sec \n", seconds() - t0); 
        }  

        // TILED COO GPU 
        else if(Opt.impType == 7){
            cout << "GPU COO has bugs! " << endl;
            MTTKRP_TILED_COO_GPU(TiledX, U, Opt);
        }

        // TILED HCSR GPU
        else if(Opt.impType == 8){

            // cout <<" Delete this function" << endl;
            // double t0 = seconds();
            // create_pointers(TiledX, Opt, 0);
            // printf("POINTERS: %.3f sec \n", seconds() - t0); 
            
            // cout <<" Uncomment the follwoing" << endl;
            // // cout << "Sorted mode: " << X.modeOrder[0] << " " << X.modeOrder[1] << " " <<X.modeOrder[2] << endl;
            create_fbrLikeSlcInds(TiledX, 0);
            create_fbrLikeSlcInds(X, Opt);
            MTTKRP_B_HCSR_GPU(TiledX, U, Opt);
        }


        // TILED + support all mode using same B-CSF
        else if(Opt.impType == 9){
            // int MTTKRPmode = 0;
            for (int MTTKRPmode = 0; MTTKRPmode < X.ndims; ++MTTKRPmode){
                randomize_mats(X, U, Opt);
                zero_mat(X, U, MTTKRPmode);  
                MTTKRP_B_HCSR_GPU_ANYMODE(TiledX, U, Opt, MTTKRPmode);
            }
        }
    }

     /* single-CSF*/
    else if(Opt.impType == 13 || Opt.impType == 14 ){

        if(Opt.verbose)
            cout << "Starting sameCSF: MTTKRP on all modes using same CSF" << endl;
        
        sort_COOtensor(X); 
        create_HCSR(X, Opt); 
        // compute_reuse(X,Opt);
        // compute_reuse_distance(X,Opt);
        /* on CPU non tiled */
        if(Opt.impType == 13){ 

            for (int MTTKRPmode = 0; MTTKRPmode < X.ndims; ++MTTKRPmode) {
                    
                randomize_mats(X, U, Opt);
                zero_mat(X, U, MTTKRPmode);  
         
                // if( MTTKRPmode == Opt.mode){
                if( X.modeOrder[0] ==  MTTKRPmode)     
                    ((X.ndims == 3) ?  MTTKRP_HCSR_CPU(X, TiledX, U, Opt) :  MTTKRP_HCSR_CPU_4D(X, U, Opt)); 
                
                // MTTKRPmode = (Opt.mode + 1) % X.ndims;
                else if( X.modeOrder[1] ==  MTTKRPmode)   {
                    create_fbrLikeSlcInds(X, Opt);
                    MTTKRP_HCSR_CPU_mode1(X, U, Opt,  MTTKRPmode);
                }
                
                // // MTTKRPmode = (Opt.mode + 2) % X.ndims;
                else if( X.modeOrder[2] ==  MTTKRPmode)  
                    MTTKRP_HCSR_CPU_mode2(X, U, Opt,  MTTKRPmode);              
            }
 
        }
       /* on GPU tiled (skipping on tiled gpu due to time constraints)*/
        if(Opt.impType == 14){ 

            int tilingMode = X.modeOrder[X.ndims -1];
            Opt.tileSize = (X.dims[tilingMode] + Opt.nTile - 1)/Opt.nTile;  
            
            if(Opt.nTile > X.dims[tilingMode]){
                cout << "Number of tiles ("<< Opt.nTile << ") should be as minimum as K's dimension (" << X.dims[tilingMode]  << "). Exiting."<< endl ;
                exit(0);
            }
            // print_HCSRtensor_4D(X);

            // split X into tiles based on K indices
            make_KTiling(X, TiledX, Opt);

            // create HCSR for each tile
            for (int tile = 0; tile < Opt.nTile; ++tile){

                if(TiledX[tile].totNnz > 0){
                    create_TiledHCSR(TiledX, Opt, tile);
                    create_fbrLikeSlcInds(TiledX, tile);
                    // print_COOtensor(X);
                    // print_TiledHCSRtensor(TiledX, tile);
                }
            }  

            // Split tiles into bins accordin to nnz in slice
            for (int tile = 0; tile < Opt.nTile; ++tile){
                if(TiledX[tile].totNnz > 0)
                    make_TiledBin(TiledX, Opt, tile);
            }
            MTTKRP_ONE_HCSR_GPU(TiledX, U, Opt);
            // MTTKRP_B_HCSR_GPU(TiledX, U, Opt);
        }
    }

    /* MI-CSF*/
    else if(Opt.impType == 11 || Opt.impType == 12){

        double t0 = seconds();

        if(Opt.verbose)
            cout << "Starting MM-CSF" << endl;

        /*Collect slice and fiber stats: Create CSF for all modes*/
        bool slcNfbrStats = true;
        int redunMode;

        Tensor *arrX = new Tensor[X.ndims]; 

        if(slcNfbrStats){

            for (int m = 0; m < X.ndims; ++m){
            
                init_tensor(arrX, X, Opt, m);

                /*crappy coding::*/

                if(m == 1){
                    if (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
                        // sameFm0m1 = true;
                        redunMode = m;
                    }
                }
                else if(m == 2){
                    if ((arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]) || 
                        (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1])){
                        // sameFm0m2 = true;
                        redunMode = m;
                    }
                }

                else if(m == 3){
                    if ((arrX[m].modeOrder[1] == arrX[m-3].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-3].modeOrder[1]) ||
                      (arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]) ||
                      (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1])){

                        redunMode = m;

                    }
                }
            
                // t0 = seconds();
                if(m !=  redunMode)
                    sort_COOtensor(arrX[m]);
                // printf("sort - mode %d - %.3f\n", m, seconds() - t0);
                
                // t0 = seconds();
                if(m !=  redunMode)
                    create_HCSR(arrX[m], Opt); 
                // printf("creat CSF - mode %d - %.3f\n", m, seconds() - t0);
                
                // get_nnzPerFiberData(arrX[m]); //merge with createCSF
                // create_hashtable(arrX[m]);
                // cout << "created Hshtable" << endl;
                // print_HCSRtensor(arrX[m]);
            }       
        }

        Opt.redunMode = redunMode;

        TiledTensor ModeWiseTiledX[X.ndims];
        t0 = seconds();
        //mm_partition_allMode(arrX, X, ModeWiseTiledX, Opt);
        mm_partition_reuseBased(arrX, X, ModeWiseTiledX, Opt);
        populate_paritions(X, ModeWiseTiledX);
        printf("mm_partition & populate - time: %.3f sec \n", seconds() - t0);
        
        t0 = seconds();
        // double start_time = omp_get_wtime();
        // omp_set_num_threads(X.ndims);
        // #pragma omp parallel 
        {
            // int threadnum = omp_get_thread_num(), numthreads = omp_get_num_threads();
            // #pragma omp for 
            for (int m = 0; m < X.ndims; ++m){
                
                if(ModeWiseTiledX[m].totNnz > 0){           
                    sort_MI_CSF(X, ModeWiseTiledX, m);
                    create_TiledHCSR(ModeWiseTiledX, Opt, m);
                    create_fbrLikeSlcInds(ModeWiseTiledX, m);
                    // make_TiledBin(ModeWiseTiledX, Opt, m);
                    // cout << "printing " << m << endl;
                    // print_TiledCOO(ModeWiseTiledX, m);
                    // print_TiledHCSRtensor(ModeWiseTiledX, m);
                    // compute_reuse(ModeWiseTiledX, Opt, m);
                }
                // cout << threadnum << " " << numthreads << endl;
            }
        }
        // double omp_time = omp_get_wtime() - start_time;

        // printf("Sort,createCSF,createFbrIND - time: %.3f sec \n", seconds() - t0);

        /* on CPU */
        if(Opt.impType == 11){ 
            
            for (int MTTKRPmode = 0; MTTKRPmode < X.ndims; ++MTTKRPmode){

                cout << "MTTKRP on mode " << MTTKRPmode  << " using MI-CSF" << endl;

                randomize_mats(X, U, Opt);
                zero_mat(X, U, MTTKRPmode); 

                for (int m = 0; m < X.ndims; ++m){

                    int mode0 = ModeWiseTiledX[m].modeOrder[0];
                    int mode1 = ModeWiseTiledX[m].modeOrder[1];
                    int mode2 = ModeWiseTiledX[m].modeOrder[2];

                    if (mode0 == MTTKRPmode && ModeWiseTiledX[m].totNnz)
                        MTTKRP_MIHCSR_CPU(ModeWiseTiledX, U, Opt, m);
                    
                    else if (mode1 == MTTKRPmode && ModeWiseTiledX[m].totNnz ){
                        // create_fbrLikeSlcInds(ModeWiseTiledX, U, Opt, m, MTTKRPmode);
                        MTTKRP_MIHCSR_CPU_FBR_ATOMICS(ModeWiseTiledX, U, Opt, m, MTTKRPmode);
                    }

                    else if (mode2 == MTTKRPmode && ModeWiseTiledX[m].totNnz )
                        MTTKRP_MIHCSR_CPU_ALL_ATOMICS(ModeWiseTiledX, U, Opt, m, MTTKRPmode);
                }
            }
        }
        
        /* on GPU */
        else if(Opt.impType == 12){ 
            MTTKRP_MIHCSR_GPU(ModeWiseTiledX, U, Opt);

        }
        // printf("MIHCSR incl CPU - time: %.3f sec \n", seconds() - t0);
    }

    else // e.g. -1 
        cout << "no MTTKRP" << endl;

    if(!Opt.outFileName.empty()){
        write_output(U, Opt.mode, Opt.outFileName);
    }

    if(Opt.correctness){
        if (Opt.impType == 1) {
            cout << "Already running COO seq on CPU!" << endl; 
            exit(0);
        }

        if(Opt.verbose && (Opt.impType == 12 || Opt.impType == 14))
            cout << "checking only the last mode. " << endl;
        // Opt.mode = 0;//X.modeOrder[2];
        // int MTTKRPmode = 2;
        Opt.mode = X.ndims-1;//((Opt.impType == 12 || Opt.impType == 14 ) ?  X.ndims-1 : Opt.mode);
        int mode = Opt.mode;
        int nr = U[mode].nRows;  
        int nc = U[mode].nCols;
        DTYPE *out = (DTYPE*)malloc(nr * nc * sizeof(DTYPE));
        memcpy(out, U[mode].vals, nr*nc * sizeof(DTYPE));
        print_matrix(U, mode);
        randomize_mats(X, U, Opt);
        zero_mat(X, U, mode);

        cout << "correctness with COO on mode " << mode <<". "<< endl;
        ((X.ndims == 3) ?  MTTKRP_COO_CPU(X, U, Opt) :  MTTKRP_COO_CPU_4D(X, U, Opt));
         // MTTKRP_HCSR_CPU_slc(X, TiledX, U, Opt);
        print_matrix(U, mode);
        correctness_check(out, U[mode].vals, nr, nc);
    }
}


