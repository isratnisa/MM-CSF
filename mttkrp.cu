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

    Tensor X;
    load_tensor(X, Opt);

    for (int i = 1; i < X.ndims; ++i)
    {
        int switchMode;

        if(X.switchBC){
            if(i == 1)
                switchMode = 2;
            else if(i == 2)
                switchMode = 1;
        }
        else
            switchMode = i;
    }

    // Opt.print();


    // compute_accessK(X, Opt);

    // create_write_heavy(X, Opt); //make a combined on for write and read

    // check if appropriate file is loaded
    string fileNameEndwith;
    // cout  << "order " <<  X.modeOrder[0] << " " <<
    // X.modeOrder[1] << " " << X.modeOrder[2] << endl;
    if(X.switchBC)
        fileNameEndwith = to_string(X.modeOrder[0]) + to_string(X.modeOrder[2]) + to_string(X.modeOrder[1]);
    else
        fileNameEndwith = to_string(X.modeOrder[0]) + to_string(X.modeOrder[1]) + to_string(X.modeOrder[2]);
    std::size_t found = Opt.inFileName.find(fileNameEndwith);
    if (found==std::string::npos){
        cout << "Not the correct file for this mode" << endl;
        exit(0);
    }
    
    //TBD:: fix hard coded 3
    Matrix U[3];   
    create_mats(X, U, Opt);
    randomize_mats(X, U, Opt);
    zero_mat(X, U, Opt.mode);

    if(Opt.verbose)
        cout << endl << "Starting MTTKRP..." << endl;  
    
    // print tensors and statistics
    if(Opt.impType == 0){
        double t0 = seconds();
        // print_COOtensor(X);
        //enable it
        create_HCSR(X, Opt);
        tensor_stats(X);
        //print_HCSRtensor(X);
        printf("COO CPU - time: %.3f sec \n", seconds() - t0);
    }
    // COO CPU   
    if(Opt.impType == 1){
        double t0 = seconds();
        MTTKRP_COO_CPU(X, U, Opt);
        printf("COO CPU - time: %.3f sec \n", seconds() - t0);
    }

    // HCSR CPU   
    else if(Opt.impType == 2){
        create_HCSR(X, Opt);   
        // print_COOtensor(X); 
        // print_HCSRtensor(X);     
        double t0 = seconds();
        MTTKRP_HCSR_CPU(X, U, Opt);
        printf("gcc no opt : HCSR CPU - time: %.3f sec \n", seconds() - t0);
    }

    // COO GPU  
    else if(Opt.impType == 3){
        cout << " GPU COO has bugs! " << endl;
        MTTKRP_COO_GPU(X, U, Opt);
    }

    // HCSR GPU  
    else if(Opt.impType == 4){
        create_HCSR(X, Opt);
        MTTKRP_HCSR_GPU(X, U, Opt);
    }
    // HYB CPU
    else if(Opt.impType == 9){
        // print_COOtensor(X);
        create_HCSR(X, Opt);
         // print_HCSRtensor(X);
        HYBTensor HybX(X);
        create_HYB(HybX, X, Opt);
        // make_CSLBin(HybX, Opt);
        make_Bin(HybX, Opt);
        // print_HYBtensor(HybX);
        MTTKRP_HYB_GPU(HybX, U, Opt);
        //MTTKRP_HYB_CPU(HybX, U, Opt);
    }
    // // HYB GPU
    // else if(Opt.impType == 10){
    //     // MTTKRP_TILED_HCSR_GPU(TiledX, U, Opt);
    // }


    // Tiled versions
    else if(Opt.impType >= 5 && Opt.impType < 9){
        
        TiledTensor TiledX[Opt.nTile];
        create_HCSR(X, Opt);
        // print_HCSRtensor(X);
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
            create_TiledHCSR(TiledX, Opt, tile);
            // print_TiledHCSRtensor(TiledX, tile);
        }  

        // Split tiles into bins accordin to nnz in slice
        for (int tile = 0; tile < Opt.nTile; ++tile){
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

    if(!Opt.outFileName.empty() || Opt.correctness)
        write_output(U, X.modeOrder[0], Opt.outFileName);

    if(Opt.correctness){
        Opt.outFileName = "nell2_tmp";
        string seqFileName = "files/nell2_seq";
        // MTTKRP_COO_CPU(X, U, Opt);
        // write_output(U, X.modeOrder[0], seqFileName);
        correctness_check(U, X.modeOrder[0], Opt.outFileName, seqFileName);
    }
}


