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
 
    Options Opt = parse_cmd_options(argc, argv);
    // Opt.print();
    Tensor X;
    load_tensor(X, Opt.inFileName);

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
        MTTKRP_HCSR_CPU(X, U, Opt.mode, Opt.R);
        printf("gcc no opt : HCSR CPU - time: %.3f sec \n", seconds() - t0);
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


