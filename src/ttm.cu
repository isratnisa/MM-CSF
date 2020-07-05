#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <utility>  
#include <math.h> 
#include <omp.h>
#include <cuda.h>
#include "ttm_cpu.h"
#include "ttm_gpu.h"
#include <bits/stdc++.h>  

using namespace std;

int main(int argc, char* argv[]){ 
 
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    Options Opt = parse_cmd_options(argc, argv);

    Tensor X;
    load_tensor(X, Opt);

    create_HCSR(X, Opt);  

    // check if appropriate file is loaded
    string fileNameEndwith;

    fileNameEndwith = to_string(X.modeOrder[0]) ;//+ to_string(X.modeOrder[1]) + to_string(X.modeOrder[2]);
    std::size_t found = Opt.inFileName.find(fileNameEndwith);
    if (found==std::string::npos){
        cout << "Not the correct file for this mode" << endl;
        exit(0);
    }
      
    Matrix *U = new Matrix[X.ndims]; 
    create_mats(X, U, Opt, false);
    randomize_mats(X, U, Opt);
    zero_mat(X, U, Opt.mode);

    //allocate space fro intermediate tensor Y (Y = X * Un)

    semiSpTensor Y;
    cout << "calling allocation " << endl;
    prepare_Y(X, Y, Opt);

    if(Opt.verbose)
        cout << endl << "Starting TTM..." << endl;  
    
    // print tensors and statistics
    if(Opt.impType == 0){
        double t0 = seconds();
        create_HCSR(X, Opt);
        tensor_stats(X);
        // print_HCSRtensor(X);
    }
    // CPU   
    if(Opt.impType == 1){
        double t0 = seconds();
        // ((X.ndims == 3) ?  TTM_COO_CPU(X, U, Opt) :  TTM_COO_CPU_4D(X, U, Opt));  
        TTM_CPU(X, Y, U, Opt); 
        printf("TTM - COO CPU time: %.3f sec \n", seconds() - t0);
    }

    // GPU  
    else if(Opt.impType == 2 || Opt.impType == 3){
        TTM_GPU(X, Y, U, Opt); 
    }
    // // HYB CPU
    // else if(Opt.impType == 9){
    //     create_HCSR(X, Opt);
    //     HYBTensor HybX(X);
    //     cout << "Creating HYB... " ;
    //     double t0 = seconds();
    //     ((X.ndims == 3) ?  create_HYB(HybX, X, Opt) :  create_HYB_4D(HybX, X, Opt));   
    //     printf("create HYB - time: %.3f sec \n", seconds() - t0);

    //     make_HybBin(HybX, Opt);
    //     // print_HYBtensor(HybX);
        
    //     // ((X.ndims == 3) ?  MTTKRP_HYB_CPU(HybX, U, Opt) :  MTTKRP_HYB_CPU_4D(HybX, U, Opt));   
    //     MTTKRP_HYB_GPU(HybX, U, Opt);
        
    // }
    // // // HYB GPU
    // // else if(Opt.impType == 10){
    // //     // MTTKRP_TILED_HCSR_GPU(TiledX, U, Opt);
    // // }


    // // Tiled versions
    // else if(Opt.impType >= 5 && Opt.impType < 9){
        
    //     TiledTensor TiledX[Opt.nTile];
    //     create_HCSR(X, Opt);
    //     // print_HCSRtensor(X);
    //     int tilingMode = X.modeOrder[X.ndims -1];
        
    //     Opt.tileSize = (X.dims[tilingMode] + Opt.nTile - 1)/Opt.nTile;
    //     if(Opt.nTile > X.dims[tilingMode]){
    //         cout << "Number of tiles ("<< Opt.nTile << ") should be as minimum as K's dimension (" << X.dims[tilingMode]  << "). Exiting."<< endl ;
    //         exit(0);
    //     }

    //     // split X into tiles based on K indices
    //     make_KTiling(X, TiledX, Opt);
        
    //     // create HCSR for each tile
    //     for (int tile = 0; tile < Opt.nTile; ++tile){
    //         create_TiledHCSR(TiledX, Opt, tile);
    //         // print_TiledHCSRtensor(TiledX, tile);
    //     }  

    //     // Split tiles into bins accordin to nnz in slice
    //     for (int tile = 0; tile < Opt.nTile; ++tile){
    //         make_TiledBin(TiledX, Opt, tile);
    //     }

    //     // COO GPU  
    //     if(Opt.impType == 5){
    //         double t0 = seconds();
    //         MTTKRP_TILED_COO_CPU(TiledX, U, Opt); 
    //         printf("TILED COO CPU - time: %.3f sec \n", seconds() - t0);  
    //     }

    //      // HCSR GPU  
    //     else if(Opt.impType == 6){
    //         double t0 = seconds();
    //         ((X.ndims == 3) ? MTTKRP_TILED_HCSR_CPU(TiledX, U, Opt) : MTTKRP_TILED_HCSR_CPU_4D(TiledX, U, Opt)); 
    //         printf("TILED HCSR CPU - time: %.3f sec \n", seconds() - t0); 
    //     }  

    //     //COO GPU 
    //     else if(Opt.impType == 7){
    //         cout << "GPU COO has bugs! " << endl;
    //         MTTKRP_TILED_COO_GPU(TiledX, U, Opt);
    //     }

    //     // HCSR GPU
    //     else if(Opt.impType == 8){
    //         MTTKRP_TILED_HCSR_GPU(TiledX, U, Opt);
    //     }
    // }
    // else // e.g. -1 
    //     cout << "no MTTKRP" << endl;

    if(!Opt.outFileName.empty()){
        cout << "Writing Y to " << Opt.outFileName << endl;
        write_output_ttmY(Y, X.modeOrder[0], Opt.outFileName);
    }

    /** Correctness check **/
    if(Opt.correctness){
        cout << "DO COO...now incorrect with fbr threshold " << endl;

        cout << "correctness with CPU " << endl;
        
        if (Opt.impType == 1) {
            cout << "Already running COO seq on CPU!" << endl; 
            exit(0);
        }
        
        int mode = Opt.mode;
        DTYPE *out = (DTYPE*)malloc(Y.nRows * Y.nCols * sizeof(DTYPE));
        memcpy(out, Y.vals, Y.nRows * Y.nCols * sizeof(DTYPE));    
        memset(Y.vals, 0, Y.nRows * Y.nCols * sizeof(DTYPE));
        // ((X.ndims == 3) ?  TTM_CPU(X, U, Opt) :  MTTKRP_COO_CPU_4D(X, U, Opt));   
        TTM_CPU(X, Y, U, Opt);
        correctness_check(out, Y.vals, Y.nRows, Y.nCols);
    }

    free_all(X, Y, U);
}


