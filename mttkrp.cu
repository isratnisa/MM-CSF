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
#include "mttkrp_gpu.h" 
using namespace std;

int main(int argc, char* argv[]){ 
 
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    Options Opt = parse_cmd_options(argc, argv);

    Tensor X;
    load_tensor(X, Opt);
    sort_COOtensor(X);
    
    TiledTensor TiledX[Opt.nTile];
      
    Matrix *U = new Matrix[X.ndims]; 
    create_mats(X, U, Opt, false);
    randomize_mats(X, U, Opt);
    zero_mat(X, U, Opt.mode);

    if(Opt.verbose)
        cout << endl << "Starting MTTKRP..." << endl;  
    
    // print tensors and statistics
    if(Opt.impType == 0){
        double t0 = seconds();
        // print_COOtensor(X);
        create_HCSR(X, Opt);
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

        /*MTTKRP on mode 0, 1 and 2 using same HCSR sorted as 0-1-2*/

        // int mode = 0;
        // cout << "Performing mode " << curMode[mode] << endl;
        
        // ((X.ndims == 3) ?  MTTKRP_HCSR_CPU(X, TiledX, U, Opt) :  MTTKRP_HCSR_CPU_4D(X, U, Opt));  
     
        // mode = 1;
        // cout << "Performing mode " << curMode[mode] << endl;
        // randomize_mats(X, U, Opt);
        // zero_mat(X, U, curMode[mode]); 
        // MTTKRP_HCSR_CPU_mode1(X, U, Opt, curMode[mode]); 
        // // write_output(U, curMode[mode], "tmp1"); 

        // mode = 2;
        // cout << "Performing mode " << curMode[mode] << endl;
        // randomize_mats(X, U, Opt);
        // zero_mat(X, U, curMode[mode]); 
        // MTTKRP_HCSR_CPU_mode2(X, TiledX, U, Opt, curMode[mode]);
        // // write_output(U, curMode[mode], "tmp2");  
        
       
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

        // make tile fit in shared
        if(Opt.impType == 9){
            Opt.tileSize = 192;
            Opt.nTile = (X.dims[tilingMode] + Opt.tileSize - 1)/Opt.tileSize;
        }
        else 
            Opt.tileSize = (X.dims[tilingMode] + Opt.nTile - 1)/Opt.nTile;  
        
        cout << "tile size and numbers: " <<  Opt.tileSize << " " << Opt.nTile << endl;
        
        if(Opt.nTile > X.dims[tilingMode]){
            cout << "Number of tiles ("<< Opt.nTile << ") should be as minimum as K's dimension (" << X.dims[tilingMode]  << "). Exiting."<< endl ;
            exit(0);
        }

        // split X into tiles based on K indices
        make_KTiling(X, TiledX, Opt);
        
        // create HCSR for each tile
        cout << "TBD:: active rows" << endl;
        for (int tile = 0; tile < Opt.nTile; ++tile){

            if(TiledX[tile].totNnz > 0)
                create_TiledHCSR(TiledX, Opt, tile);
            // print_TiledHCSRtensor(TiledX, tile);
        }  

        // Split tiles into bins accordin to nnz in slice
        for (int tile = 0; tile < Opt.nTile; ++tile){
            if(TiledX[tile].totNnz > 0)
                make_TiledBin(TiledX, Opt, tile);
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
            MTTKRP_TILED_HCSR_GPU(TiledX, U, Opt);
        }


        // TILED + shared HCSR GPU
        else if(Opt.impType == 9){
            MTTKRP_TILED_HCSR_GPU(TiledX, U, Opt);
        }
    }

    /* MI-CSF*/

    else if(Opt.impType == 11){

        cout << "Development phase- CPU done" << endl;
        cout << "Starting MI-CSF" << endl;
        sort_COOtensor(X);
        TiledTensor ModeWiseTiledX[X.ndims];
        find_hvyslc_allMode(X, ModeWiseTiledX);
        // print_COOtensor( X);

        int *curMode = new int [X.ndims];    
        
        for (int m = 0; m < X.ndims; ++m){

            curMode[m] = (m + Opt.mode) % X.ndims; 
            
            if(ModeWiseTiledX[m].totNnz > 0){           
                sort_MI_CSF(X, ModeWiseTiledX, m);
                create_TiledHCSR(ModeWiseTiledX, Opt, m);
                // print_TiledHCSRtensor(ModeWiseTiledX, m);
            }
        }

        // cout << "MTTKRP on mode 0 using MI-CSF" << endl;
        // MTTKRP_MIHCSR_CPU(ModeWiseTiledX, U, Opt, 0);
        // MTTKRP_MIHCSR_CPU_mode0_using201(ModeWiseTiledX, U, Opt, 2);

        cout << "MTTKRP on mode 2 using MI-CSF" << endl;
        int mode = 2;
        randomize_mats(X, U, Opt);
        zero_mat(X, U, mode); 
        MTTKRP_MIHCSR_CPU(ModeWiseTiledX, U, Opt, mode);
        MTTKRP_MIHCSR_CPU_mode2_using012(ModeWiseTiledX, U, Opt, 0);
        // write_output(U, curMode[mode], "tmp2");  
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
        cout << "!!!CHECK CORRECTNESS MODE!!!" << endl;
        int mode = Opt.mode;
        int nr = U[mode].nRows;  
        int nc = U[mode].nCols;
        DTYPE *out = (DTYPE*)malloc(nr * nc * sizeof(DTYPE));
        memcpy(out, U[mode].vals, nr*nc * sizeof(DTYPE));
        zero_mat(X, U, Opt.mode);
        cout << "correctness with COO " << endl;
        ((X.ndims == 3) ?  MTTKRP_COO_CPU(X, U, Opt) :  MTTKRP_COO_CPU_4D(X, U, Opt));   
        correctness_check(out, U[mode].vals, nr, nc);
    }
}


