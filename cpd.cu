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
    Opt.doCPD = true;
    
    Tensor X;
    load_tensor(X, Opt);
    // sort_COOtensor(X);
    // check_opt(X, Opt); //check options are good
    // MPI_param MPIparam;
    
    TiledTensor TiledX[Opt.nTile];
      
    Matrix *U = new Matrix[X.ndims]; 
    create_mats(X, U, Opt, false);
    randomize_mats(X, U, Opt);

    // /* Preprocessing of the tensor before starting CPD*/
    // // HCSR CPU  
    // create_HCSR(X, Opt);    

    // // HYB CPU
    // if(Opt.impType == 10){
        
    //     HYBTensor HybX(X);
    //     ((X.ndims == 3) ?  create_HYB(HybX, X, Opt) :  create_HYB_4D(HybX, X, Opt));   
    //     make_HybBin(HybX, Opt);  
    //     // MTTKRP_HYB_GPU(HybX, U, Opt);      
    // }

    // /* Tiled versions */
    // else if(Opt.impType >= 5 && Opt.impType < 10){

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
    //         if(TiledX[tile].totNnz > 0)
    //             create_TiledHCSR(TiledX, Opt, tile);
    //     }  

    //     // Split tiles into bins accordin to nnz in slice
    //     for (int tile = 0; tile < Opt.nTile; ++tile){
    //         if(TiledX[tile].totNnz > 0)
    //             make_TiledBin(TiledX, Opt, tile);
    //     }
    //     // TILED HCSR GPU
    //     if(Opt.impType == 8){
        
    //         create_fbrLikeSlcInds(TiledX, 0);
    //         create_fbrLikeSlcInds(X, Opt);
    //         // MTTKRP_B_HCSR_GPU(TiledX, U, Opt);
    //     }
    // }

    //  /* ONE-CSF*/
    // else if(Opt.impType == 13 || Opt.impType == 14 ){

    //     if(Opt.verbose)
    //         cout << "Starting ONE-CSF: " << endl;
        
    //    /* on GPU tiled (skipping on tiled gpu due to time constraints)*/
    //     if(Opt.impType == 14){ 

    //         int tilingMode = X.modeOrder[X.ndims -1];
    //         Opt.tileSize = (X.dims[tilingMode] + Opt.nTile - 1)/Opt.nTile;  

    //         // split X into tiles based on K indices
    //         make_KTiling(X, TiledX, Opt);

    //         // create HCSR for each tile
    //         for (int tile = 0; tile < Opt.nTile; ++tile){

    //             if(TiledX[tile].totNnz > 0){
    //                 create_TiledHCSR(TiledX, Opt, tile);
    //                 create_fbrLikeSlcInds(TiledX, tile);
    //             }
    //         }  

    //         // Split tiles into bins accordin to nnz in slice
    //         for (int tile = 0; tile < Opt.nTile; ++tile){
    //             if(TiledX[tile].totNnz > 0)
    //                 make_TiledBin(TiledX, Opt, tile);
    //         }
    //         // MTTKRP_ONE_HCSR_GPU(TiledX, U, Opt);
    //     }
    // }

    // /* MM-CSF*/
    // else if(Opt.impType == 11 || Opt.impType == 12){

    //     double t0 = seconds();

    //     if(Opt.verbose)
    //         cout << "Starting MI-CSF" << endl;

    //     /*Collect slice and fiber stats: Create CSF for all modes*/
    //     bool slcNfbrStats = true;

    //     Tensor *arrX = new Tensor[X.ndims]; 

    //     if(slcNfbrStats){

    //         for (int m = 0; m < X.ndims; ++m){         
    //             init_tensor(arrX, X, Opt, m);
    //             if(m!= Opt.mode) //already sorted
    //                 sort_COOtensor(arrX[m]);
    //             create_HCSR(arrX[m], Opt); 
    //         }       
    //     }

    //     TiledTensor ModeWiseTiledX[X.ndims];
    //     t0 = seconds();
    //     //mm_partition_allMode(arrX, X, ModeWiseTiledX, Opt);
    //     mm_partition_reuseBased(arrX, X, ModeWiseTiledX, Opt);
    //     populate_paritions(X, ModeWiseTiledX);
        
    //     omp_set_num_threads(X.ndims);
    //     // #pragma omp parallel 
    //     {
    //         // #pragma omp for 
    //         for (int m = 0; m < X.ndims; ++m){
                
    //             if(ModeWiseTiledX[m].totNnz > 0){           
    //                 sort_MI_CSF(X, ModeWiseTiledX, m);
    //                 create_TiledHCSR(ModeWiseTiledX, Opt, m);
    //                 create_fbrLikeSlcInds(ModeWiseTiledX, m);
    //                 make_TiledBin(ModeWiseTiledX, Opt, m);
    //             }
    //         }
    //     }
    //     // MTTKRP_MIHCSR_GPU(ModeWiseTiledX, U, Opt);
    // }
    
    if(Opt.verbose)
        cout << endl << "Starting CPD..." << endl;  

    DTYPE * lambda = (DTYPE*) malloc(U[0].nCols * sizeof(DTYPE));

    /* Start CPD here*/
    double cpd_t0 = seconds();
    cpd(X, U, Opt, lambda);
    printf("COO CPU CPD time: %.3f sec \n", seconds() - cpd_t0);
    free(lambda);

    if(Opt.correctness){
        if (Opt.impType == 1) {
            cout << "Already running COO seq on CPU!" << endl; 
            exit(0);
        }
        if(Opt.verbose && (Opt.impType == 12 || Opt.impType == 14))
            cout << "checking only the last mode. " << endl;
        // Opt.mode = 0;//X.modeOrder[2];
        Opt.mode = ((Opt.impType == 12 || Opt.impType == 14 ) ? X.ndims-1 : Opt.mode);
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


