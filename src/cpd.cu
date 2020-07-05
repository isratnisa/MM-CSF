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
    Opt.impType = 12;
    
    Tensor X;
    load_tensor(X, Opt);
    // sort_COOtensor(X);
    // check_opt(X, Opt); //check options are good
    // MPI_param MPIparam;
    
    TiledTensor TiledX[Opt.nTile];
    TiledTensor ModeWiseTiledX[X.ndims];
      
    Matrix *U = new Matrix[X.ndims]; 
    create_mats(X, U, Opt, false);
    randomize_mats(X, U, Opt);

    /* Preprocessing of the tensor before starting CPD*/
    // HCSR CPU  
    create_HCSR(X, Opt);    

    /* Tiled versions */

    /* MM-CSF*/
    if(Opt.impType == 11 || Opt.impType == 12){

        double t0 = seconds();

        if(Opt.verbose)
            cout << "Starting MI-CSF" << endl;

        /*Collect slice and fiber stats: Create CSF for all modes*/
        bool slcNfbrStats = true;

        Tensor *arrX = new Tensor[X.ndims]; 

        if(slcNfbrStats){

            for (int m = 0; m < X.ndims; ++m){         
                init_tensor(arrX, X, Opt, m);
                // if(m!= Opt.mode) //already sorted
                sort_COOtensor(arrX[m]);
                create_HCSR(arrX[m], Opt); 
            }       
        }

        // TiledTensor ModeWiseTiledX[X.ndims];
        t0 = seconds();
        //mm_partition_allMode(arrX, X, ModeWiseTiledX, Opt);
        mm_partition_reuseBased(arrX, X, ModeWiseTiledX, Opt);
        populate_paritions(X, ModeWiseTiledX);
        
        // #pragma omp parallel 
        {
            // #pragma omp for 
            for (int m = 0; m < X.ndims; ++m){
                
                if(ModeWiseTiledX[m].totNnz > 0){           
                    sort_MI_CSF(X, ModeWiseTiledX, m);
                    create_TiledHCSR(ModeWiseTiledX, Opt, m);
                    create_fbrLikeSlcInds(ModeWiseTiledX, m);
                    make_TiledBin(ModeWiseTiledX, Opt, m);
                }
            }
        }
        // MTTKRP_MIHCSR_GPU(ModeWiseTiledX, U, Opt);
    }
    
    if(Opt.verbose)
        cout << endl << "Starting CPD..." << endl;  

    DTYPE * lambda = (DTYPE*) malloc(U[0].nCols * sizeof(DTYPE));

    /* Start CPD here*/
    double cpd_t0 = seconds();
    cpd(X, ModeWiseTiledX, U, Opt, lambda);
    printf("COO CPU CPD time: %.3f sec \n", seconds() - cpd_t0);
    free(lambda);
}


