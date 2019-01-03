#ifndef UTIL_H
#define UTIL_H

#define DTYPE float
#define ITYPE size_t

#include <vector>
#include <algorithm>
#include <iterator>
#include <utility>  
#include <bits/stdc++.h>  
#include <time.h>
#include <sys/time.h>
#include <iomanip> 
#include <iostream>
#include <omp.h>

using namespace std;

class Tensor{
    public:
        ITYPE ndims;
        ITYPE *dims;
        ITYPE totNnz;
        ITYPE nFibers;
        ITYPE *accessK;
        bool switchBC = false; // if true change matrix rand() to 1
        std::vector<ITYPE> modeOrder;
        std::vector<vector<ITYPE>> inds;
        std::vector<DTYPE> vals;
        std::vector<vector<ITYPE>> fbrPtr;
        std::vector<vector<ITYPE>> fbrIdx;
        std::vector<vector<ITYPE>> slcMapperBin;
        std::vector<vector<ITYPE>> rwBin;
};

class HYBTensor{
    public:
        ITYPE ndims;
        ITYPE *dims;
        ITYPE totNnz;
        ITYPE HCSRnnz;
        ITYPE COOnnz;
        ITYPE CSLnnz;
        ITYPE nFibers;
        ITYPE *accessK;
        std::vector<ITYPE> modeOrder;
        ITYPE **inds;
        DTYPE *vals;
        std::vector<vector<ITYPE>> fbrPtr;
        std::vector<vector<ITYPE>> fbrIdx;
        std::vector<vector<ITYPE>> slcMapperBin;
        ITYPE **COOinds;
        DTYPE *COOvals;
        std::vector<ITYPE> CSLslicePtr;
        std::vector<ITYPE> CSLsliceIdx;
        ITYPE **CSLinds;
        DTYPE *CSLvals;
        std::vector<vector<ITYPE>> CSLslcMapperBin;
        
        HYBTensor(const Tensor &X) 
        { 
            ndims = X.ndims;
            dims = new ITYPE[X.ndims];
            totNnz = X.totNnz;
            for (int i = 0; i < ndims; ++i)
            {
                dims[i] = X.dims[i];
                modeOrder.push_back(X.modeOrder[i]);
            }
        } 
};

class TiledTensor{
    public:
        ITYPE ndims;
        ITYPE *dims;
        ITYPE totNnz;
        ITYPE nFibers;
        ITYPE *accessK;
        std::vector<ITYPE> modeOrder;
        std::vector<vector<ITYPE>> inds;
        std::vector<DTYPE> vals;
        std::vector<vector<ITYPE>> fbrPtr;
        std::vector<vector<ITYPE>> fbrIdx;
        std::vector<vector<ITYPE>> slcMapperBin;
        std::vector<vector<ITYPE>> rwBin;
};

class Matrix{
    public:
        ITYPE nRows;
        ITYPE nCols;
        DTYPE *vals;
};

class semiSpTensor{
    public:
        ITYPE nRows;
        ITYPE nCols;
        DTYPE *vals;
};

class Options {
public:
    ITYPE R = 32;
    ITYPE mode = 0;
    ITYPE impType = 1;
    ITYPE warpPerSlice = 4;
    ITYPE nTile = 1;
    ITYPE tileSize;
    bool verbose = false;     // if true change matrix rand() to 1
    bool correctness = false; 
    string inFileName; 
    string outFileName; 
    ITYPE nBin = 10;
    ITYPE fbrThreashold = 99999999;

    void print() {
        std::cout << "R = " << R << '\n';
        std::cout << "mode = " << mode << '\n';
        std::cout << "impType = " << impType << '\n';
        std::cout << "warpPerSlice = " << warpPerSlice << '\n';
        std::cout << "nTiles = " << nTile << '\n';
        std::cout << "verbose = " << verbose << '\n';

        // must provide input file name 
        if(inFileName.empty()){
            cout << "Provide input file path. Program will exit." << endl;
            exit(0);
        }
        else{
            std::cout << "input file name = " << inFileName << '\n';
        }

        if(!outFileName.empty())
            std::cout << "output file name = " << outFileName << '\n';

    }
};

inline int load_tensor(Tensor &X, const Options &Opt){
 
    //cout << endl << "Loading tensor.." << endl;   
    string filename = Opt.inFileName;
    ITYPE index;
    DTYPE vid=0;

    ITYPE switchMode = 0;
    bool switchBC =  false;

    ifstream fp(filename); 

    if(fp.fail()){
        cout << filename << " does not exist!" << endl;
        exit(0);
    }

    fp >> X.ndims; 

    X.dims = new ITYPE[X.ndims];

    for (int i = 0; i < X.ndims; ++i){
        // mode 0 never switches
        fp >> X.dims[i];      
        X.inds.push_back(std::vector<ITYPE>());
    }

    // fix it:: hard coded for 3D tensor
    int mode1 = (1 + Opt.mode) % X.ndims;   
    int mode2 = (2 + Opt.mode) % X.ndims;
    if( X.dims[mode1] > X.dims[mode2]) switchBC = true;

    for (int i = 0; i < X.ndims; ++i){
        
        // mode 0 never switches
        if(i > 0 && switchBC){

            if(i == 1)
                switchMode = 2;
            else if(i == 2)
                switchMode = 1;
        }
        else
            switchMode = i;       
        X.modeOrder.push_back((switchMode + Opt.mode) % X.ndims);
    }
     // if(switchBC) std::swap(X.dims[1], X.dims[2]);

    while(fp >> index) {
        X.inds[0].push_back(index-1);
        for (int i = 1; i < X.ndims; ++i)
        {      
            fp >> index;
            X.inds[i].push_back(index-1);   
        }
        fp >> vid;
        X.vals.push_back(vid);

    }
    X.totNnz = X.vals.size();

    return 0;
}

inline bool sort_pred(tuple <ITYPE, ITYPE, ITYPE, DTYPE> left, 
                  tuple <ITYPE, ITYPE, ITYPE, DTYPE> right) {
    // return get<0>(left) < get<0>(right);

    if (get<0>(left) != get<0>(right)) 
        return (get<0>(left) < get<0>(right));
    
    if (get<1>(left) != get<1>(right)) 
        return (get<1>(left) < get<1>(right));
      
    return (get<2>(left) < get<2>(right));
}

inline int sort_COOtensor(Tensor &X){

    const ITYPE mode0 = X.modeOrder[0];
    const ITYPE mode1 = X.modeOrder[1];
    const ITYPE mode2 = X.modeOrder[2];

    vector < tuple <ITYPE, ITYPE, ITYPE, DTYPE> > items;
    tuple <ITYPE, ITYPE, ITYPE, DTYPE> ap;

    for (long idx = 0; idx < X.totNnz; ++idx) { 

        ap=std::make_tuple(X.inds[mode0][idx], X.inds[mode1][idx], X.inds[mode2][idx], X.vals[idx]); 
        items.push_back(ap);
    }

    sort(items.begin(), items.end(), sort_pred);

    for (long idx = 0; idx < X.totNnz; ++idx) {

            X.inds[mode0][idx] = get<0>(items[idx]);
            X.inds[mode1][idx] = get<1>(items[idx]);
            X.inds[mode2][idx] = get<2>(items[idx]);
            X.vals[idx] = get<3>(items[idx]);
    }
    
    // ofstream of("sorted.txt");

    // of << X.ndims << endl;
    // of << std::fixed;
    
    // for (int i = 0; i < X.ndims; ++i)
    // {
    //    of << X.dims[i] << " ";
    // }
    // of << endl;

    // for (long idx = 0; idx < X.totNnz; ++idx) {

    //     for (int i = 0; i < X.ndims; ++i)
    //          of << X.inds[i][idx] << " ";
    //     of << std::setprecision(1) << X.vals[idx] << endl;
    // }
}

inline int sort_MI_CSF(const Tensor &X, TiledTensor *MTX, int m){

    const ITYPE mode0 = MTX[m].modeOrder[0];
    const ITYPE mode1 = MTX[m].modeOrder[1];
    const ITYPE mode2 = MTX[m].modeOrder[2];

    vector < tuple <ITYPE, ITYPE, ITYPE, DTYPE> > items;
    tuple <ITYPE, ITYPE, ITYPE, DTYPE> ap;

    for (long idx = 0; idx < MTX[m].totNnz; ++idx) { 

        ap=std::make_tuple(MTX[m].inds[mode0][idx], MTX[m].inds[mode1][idx], MTX[m].inds[mode2][idx], MTX[m].vals[idx]); 
        items.push_back(ap);
    }

    sort(items.begin(), items.end(), sort_pred);

    for (long idx = 0; idx < MTX[m].totNnz; ++idx) {

        MTX[m].inds[mode0][idx] = get<0>(items[idx]);
        MTX[m].inds[mode1][idx] = get<1>(items[idx]);
        MTX[m].inds[mode2][idx] = get<2>(items[idx]);
        MTX[m].vals[idx] = get<3>(items[idx]);
    }

    // cout << "sorted tile : " << m << endl;
    // for (long idx = 0; idx < MTX[m].totNnz; ++idx) { 
    // std::cout << MTX[m].inds[0][idx] << " "
    //           << MTX[m].inds[1][idx] << " "
    //           << MTX[m].inds[2][idx] << " "
    //           << MTX[m].vals[idx] <<  std::endl;
    // }
}

inline int print_COOtensor(const Tensor &X){

    cout << "Tensor X in COO format: " << endl;

    for(ITYPE x = 0; x < X.totNnz; ++x) {
        for (int i = 0; i < X.ndims; ++i)
            cout << X.inds[i][x] << " ";
        cout << X.vals[x]<< endl;
    }           

}

inline int print_TiledCOOtensor(const TiledTensor *TiledX, const int nTile){

    for (int tile = 0; tile < nTile; ++tile){
        cout << "tile: " << tile << endl;
        for(ITYPE x = 0; x < TiledX[tile].totNnz; ++x) {
            for (int i = 0; i < TiledX[tile].ndims; ++i)
                cout << TiledX[tile].inds[i][x] << " ";
            cout << endl;
        } 
    }          

}

inline int print_HCSRtensor(const Tensor &X){

    cout << "no of fibers " << X.fbrPtr[1].size() << endl;

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];

    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
        
        for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){        

            for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {
                if(mode0 == 0)
                    cout << X.fbrIdx[0][slc] << " " << X.fbrIdx[1][fbr] << " " << X.inds[2][x] << endl;
                if(mode0 == 1)
                    cout  << X.fbrIdx[1][fbr] << " " << X.inds[1][x] << " "<< X.fbrIdx[0][slc] << endl;
                if(mode0 == 2)
                    cout  << X.inds[0][x]  << " " << X.fbrIdx[0][slc] << " " << X.fbrIdx[1][fbr]<< endl;

            }            
        }
    }
}

inline int print_HCSRtensor_4D(const Tensor &X){

    cout << "no of fibers " << X.fbrPtr[1].size() << endl;

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE mode3 = X.modeOrder[3];

    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
        
        for (int fbrS = X.fbrPtr[0][slc]; fbrS < X.fbrPtr[0][slc+1]; ++fbrS){   

            for (int fbr = X.fbrPtr[1][fbrS]; fbr < X.fbrPtr[1][fbrS+1]; ++fbr){        
     
                for(ITYPE x = X.fbrPtr[2][fbr]; x < X.fbrPtr[2][fbr+1]; ++x) {
                    
                    if(mode0 == 0)
                        cout << X.fbrIdx[0][slc] << " " << X.fbrIdx[1][fbrS] << " " << X.fbrIdx[2][fbr] << " " << X.inds[3][x] << endl;
                    // if(mode0 == 1)
                    //     cout  << X.fbrIdx[1][fbr] << " " << X.inds[1][x] << " "<< X.fbrIdx[0][slc]; << endl;
                    // if(mode0 == 2)
                    //     cout  << X.inds[0][x]  << " " << X.fbrIdx[0][slc]; << " " << X.fbrIdx[1][fbr]<< endl;

                }  
            }          
        }
    }
}

inline int print_HYBtensor(const HYBTensor &HybX){

    cout << "COO " << HybX.COOnnz << endl; 

    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];

    for(ITYPE x = 0; x < HybX.COOnnz; ++x) {
    
        cout << HybX.COOinds[mode0][x] << " " << HybX.COOinds[mode1][x] << " " << HybX.COOinds[mode2][x] << endl;

    } 

    cout << "CSL " << HybX.CSLsliceIdx.size() << endl;

    for(ITYPE slc = 0; slc < HybX.CSLsliceIdx.size(); ++slc) {

        ITYPE idx0 = HybX.CSLsliceIdx[slc];
        printf("slc st- end: %d %d %d \n", slc, HybX.CSLslicePtr[slc], HybX.CSLslicePtr[slc+1] );
        for (int fbr = HybX.CSLslicePtr[slc]; fbr < HybX.CSLslicePtr[slc+1]; ++fbr){        
            printf("fbr %d :  ", fbr );   
            cout << idx0 << " " << HybX.CSLinds[mode1][fbr] << " " << HybX.CSLinds[mode2][fbr] << endl;
        }
    }
    
    cout << "HCSR " <<HybX.fbrIdx[0].size() << endl;

    for(ITYPE slc = 0; slc < HybX.fbrIdx[0].size(); ++slc) {

        ITYPE idx0 = HybX.fbrIdx[0][slc];
        int fb_st = HybX.fbrPtr[0][slc];
        int fb_end = HybX.fbrPtr[0][slc+1];
        printf("slc st- end: %d %d %d \n", slc, fb_st, fb_end );
        
        for (int fbr = fb_st; fbr < fb_end; ++fbr){        
             printf("fbr %d :  ", fbr );    
            ITYPE idx1 = HybX.fbrIdx[1][fbr];
            for(ITYPE x = HybX.fbrPtr[1][fbr]; x < HybX.fbrPtr[1][fbr+1]; ++x) {
                if(mode0 == 0)
                    cout << idx0 << " " << idx1 << " " << HybX.inds[2][x] << endl;
                if(mode0 == 1)
                    cout  << idx1 << " " << HybX.inds[1][x] << " "<< idx0 << endl;
                if(mode0 == 2)
                    cout  << HybX.inds[0][x]  << " " << idx0 << " " << idx1<< endl;

            }
        }
    }
}

inline int print_TiledHCSRtensor(TiledTensor *TiledX, int tile){

    cout << "Tile " << tile << " of Tensor X in Tiled HCSR format: " << endl;

    const ITYPE mode0 = TiledX[tile].modeOrder[0];
    const ITYPE mode1 = TiledX[tile].modeOrder[1];
    const ITYPE mode2 = TiledX[tile].modeOrder[2];
    
    for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {

        ITYPE idx0 = TiledX[tile].fbrIdx[0][slc]; //slc
        int fb_st = TiledX[tile].fbrPtr[0][slc];
        int fb_end = TiledX[tile].fbrPtr[0][slc+1];
        // printf("slc st- end: %d %d %d \n", slc, fb_st, fb_end );
        
        for (int fbr = fb_st; fbr < fb_end; ++fbr){        
            // printf("fbr %d :  ", fbr );    
            for(ITYPE x = TiledX[tile].fbrPtr[1][fbr]; x < TiledX[tile].fbrPtr[1][fbr+1]; ++x) {
                cout << idx0 << " " << TiledX[tile].inds[mode1][x] << " " << TiledX[tile].inds[mode2][x] << endl;

            }            
        }
    }
}

inline int make_KTiling(const Tensor &X, TiledTensor *TiledX, const Options &Opt){

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE mode3 = ((X.ndims == 4) ? X.modeOrder[3] : 0) ;
    
    // cout << "TBD:: get rid of dims, mode, etc. for each tile";
    for (int tile = 0; tile < Opt.nTile; ++tile){
        TiledX[tile].ndims = X.ndims;
        TiledX[tile].dims = new ITYPE[TiledX[tile].ndims]; 
        
        for (int i = 0; i < X.ndims; ++i){
            TiledX[tile].inds.push_back(std::vector<ITYPE>()); 
            TiledX[tile].dims[i] = X.dims[i];
            TiledX[tile].modeOrder.push_back(X.modeOrder[i]);
        }           
    }

    int tile = 0;

    for (int idx = 0; idx < X.totNnz; ++idx){

        tile = ((TiledX[0].ndims == 3) ? X.inds[mode2][idx]/Opt.tileSize : X.inds[mode3][idx]/Opt.tileSize) ;

        for (int i = 0; i < X.ndims; ++i)  {
            TiledX[tile].inds[i].push_back(X.inds[i][idx]); 
        }

        TiledX[tile].vals.push_back(X.vals[idx]);      
    }
    for (int tile = 0; tile < Opt.nTile; ++tile){
        TiledX[tile].totNnz = TiledX[tile].vals.size();
    }

    // Debug
    // for (int tile = 0; tile < Opt.nTile; ++tile){
    //     cout << "tile no: " << tile << endl;
        
    //     for (int d = 0; d < TiledX[tile].vals.size(); ++d){
    //         cout << TiledX[tile].inds[0][d] << " " << TiledX[tile].inds[1][d] 
    //         <<" " << TiledX[tile].inds[2][d] ;
    //         cout << endl;  
    //     }      
    //     cout << endl;     
    // }
}

inline int create_HCSR(Tensor &X, const Options &Opt){

    ITYPE fbrThreashold = Opt.fbrThreashold;

    for (int i = 0; i < X.ndims - 1; ++i){
        X.fbrPtr.push_back(std::vector<ITYPE>());
        X.fbrIdx.push_back(std::vector<ITYPE>());
    }
    
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    // ITYPE mode3 = X.modeOrder[3];

    std::vector<ITYPE> prevId(X.ndims-1);
    std::vector<ITYPE> fbrId(X.ndims-1);

    for (int i = 0; i < X.ndims-1; ++i){
        prevId[i] =  X.inds[X.modeOrder[i]][0];
        X.fbrPtr[i].push_back(0);
        X.fbrIdx[i].push_back(prevId[i]);
    }
    
    int idx = 1 ;
    
    while(idx < X.totNnz) {

        for (int i = 0; i < X.ndims-1; ++i) 
            fbrId[i] = X.inds[X.modeOrder[i]][idx];
   
        ITYPE fiberNnz = 1;
        bool sameFbr = true;

        for (int i = 0; i < X.ndims-1; ++i) {
            if(fbrId[i] != prevId[i])
                sameFbr = false;
        }
        /* creating last fiber consisting all nonzeroes in same fiber */
        while( sameFbr && idx < X.totNnz && fiberNnz < fbrThreashold){
            ++idx;
            fiberNnz++;
            for (int i = 0; i < X.ndims-1; ++i) {
                fbrId[i] = X.inds[X.modeOrder[i]][idx];   
                if(fbrId[i] != prevId[i])
                    sameFbr = false;
            }
        }

        if(idx == X.totNnz)
            break;

        /* X.ndims-2 is the last fiber ptr. Out of prev while loop means it is a new fiber. */
        X.fbrPtr[X.ndims-2].push_back(idx);
        X.fbrIdx[X.ndims-2].push_back(fbrId[X.ndims-2]);

        /* populating slice ptr and higher ptrs */
        for (int i = X.ndims - 3; i > -1 ; --i) {
            
            /* each dimension checks whether all parent/previous dimensions are in same slice/fiber/... */
            bool diffFbr = false;            
            int iDim = i;
            while(iDim > -1){
                if( fbrId[iDim] != prevId[iDim]) {//not else ..not become this in loop          
                    diffFbr = true;
                } 
                iDim--;
            }
            if(diffFbr){
                X.fbrIdx[i].push_back(fbrId[i]);
                X.fbrPtr[i].push_back((ITYPE)(X.fbrPtr[i+1].size()) - 1);
            }
        }
     
        for (int i = 0; i < X.ndims-1; ++i)
            prevId[i] =  fbrId[i];

        ++idx;
        fiberNnz = 1;
    }
    X.fbrPtr[X.ndims-2].push_back(idx);
    X.fbrIdx[X.ndims-2].push_back(fbrId[X.ndims-2]);

    for (int i = X.ndims - 3; i > -1 ; --i)
        X.fbrPtr[i].push_back((ITYPE)(X.fbrPtr[i+1].size() - 1 ));
    
    X.nFibers = X.fbrPtr[1].size() - 1;

    return 0;
}

inline int create_HYB(HYBTensor &HybX, const Tensor &X, const Options &Opt){

    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];

    ITYPE fbrThreashold = Opt.fbrThreashold;

   
    // reserving size 
    std::vector<int> arSlcNnz(X.fbrIdx[0].size(), 0);
    std::vector<bool> arFbrLenOne(X.fbrIdx[0].size(), true);
    std::vector<int> arFbrPtr(X.fbrPtr[1].size(), 0);

    ITYPE fiberNnz = 0;
    int indSize = 0, CSLindSize = 0, COOindSize = 0, curIndSize = 0, curCSLIndSize = 0, curCOOIndSize = 0;

    // #pragma omp parallel 
    {
        // #pragma omp for private(fiberNnz) reduction(+:indSize, CSLindSize, COOindSize)
    
        for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
            arFbrLenOne[slc] = true;  

            arFbrPtr[slc] =  X.fbrPtr[0][slc+1] -  X.fbrPtr[0][slc];
            for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){  
                
                fiberNnz = X.fbrPtr[1][fbr+1] - X.fbrPtr[1][fbr];   
                arSlcNnz[slc] += fiberNnz;
                if(fiberNnz > 1) 
                    arFbrLenOne[slc] = false;  
            }
            if( arSlcNnz[slc] == 1)
                COOindSize += arSlcNnz[slc];
            else if(arFbrLenOne[slc])
                CSLindSize += arSlcNnz[slc];
            else 
                indSize += arSlcNnz[slc];
        }
    }

    for (int i = 0; i < X.ndims - 1; ++i){
        HybX.fbrPtr.push_back(std::vector<ITYPE>());
        HybX.fbrIdx.push_back(std::vector<ITYPE>());
    }

    // allocating COO space
    HybX.COOinds = (ITYPE **)malloc(sizeof(ITYPE *) * HybX.ndims);
    HybX.CSLinds = (ITYPE **)malloc(sizeof(ITYPE *) * HybX.ndims);
    HybX.inds = (ITYPE **)malloc(sizeof(ITYPE *) * HybX.ndims);

    for(int m = 0; m < HybX.ndims; m++)
    {
        HybX.COOinds[m] = (ITYPE (*))malloc(COOindSize * sizeof(ITYPE));
    }
    HybX.COOvals = (DTYPE*)malloc( COOindSize * sizeof(DTYPE));
    
    HybX.inds[mode2] = (ITYPE *)malloc(indSize * sizeof(ITYPE));
    HybX.vals = (DTYPE*)malloc( indSize * sizeof(DTYPE));
    
    HybX.CSLinds[mode1] = (ITYPE *)malloc(CSLindSize * sizeof(ITYPE));
    HybX.CSLinds[mode2] = (ITYPE *)malloc(CSLindSize * sizeof(ITYPE));
    HybX.CSLvals = (DTYPE *)malloc(CSLindSize * sizeof(DTYPE));

    int usedCOOFbr = 0, usedCSLFbr = 0, usedHCSRFbr = 0;
    curCOOIndSize = 0; curIndSize = 0; curCSLIndSize = 0;
    
    
    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

        // cout << slc <<" slcnnz " << arSlcNnz[slc] <<  endl;

        const int stFiber =  X.fbrPtr[0][slc];
        const int endFiber =  X.fbrPtr[0][slc+1];
        const int idx_st = X.fbrPtr[1][stFiber];
        const int idx_end = X.fbrPtr[1][endFiber];
        
        if(arSlcNnz[slc] == 1){    
            // replace with array
            int idx =  X.fbrPtr[1][X.fbrPtr[0][slc]];

            HybX.COOinds[mode0][curCOOIndSize] = (X.fbrIdx[0][slc]);
            HybX.COOinds[mode1][curCOOIndSize] = (X.fbrIdx[1][stFiber]);
            HybX.COOinds[mode2][curCOOIndSize] = (X.inds[mode2][idx]); 
            HybX.COOvals[curCOOIndSize] = (X.vals[idx]);  
            curCOOIndSize++; 
            
            usedCOOFbr++;
        
        }
        else if(arFbrLenOne[slc] == true) { 

            HybX.CSLslicePtr.push_back(X.fbrPtr[0][slc] - (usedCOOFbr + usedHCSRFbr));
            HybX.CSLsliceIdx.push_back(X.fbrIdx[0][slc]);                  
            
            memcpy(&(HybX.CSLinds[mode1][0]) + curCSLIndSize, &(X.fbrIdx[1][0]) + stFiber, sizeof(ITYPE) * arSlcNnz[slc]);
            memcpy(&(HybX.CSLinds[mode2][0]) + curCSLIndSize, &(X.inds[mode2][0]) + idx_st, sizeof(ITYPE) * arSlcNnz[slc]);
            memcpy(&(HybX.CSLvals[0]) + curCSLIndSize, &(X.vals[0]) + idx_st, sizeof(DTYPE) * arSlcNnz[slc]);
            
            curCSLIndSize += arSlcNnz[slc];
            usedCSLFbr +=  X.fbrPtr[0][slc + 1] - X.fbrPtr[0][slc];
        }
        else{

            HybX.fbrPtr[0].push_back(X.fbrPtr[0][slc] - (usedCOOFbr + usedCSLFbr));
            HybX.fbrIdx[0].push_back(X.fbrIdx[0][slc]);

            int nfiber = X.fbrPtr[0][slc+1] -  X.fbrPtr[0][slc];   
            
            for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){   

                HybX.fbrPtr[1].push_back(X.fbrPtr[1][fbr] - (usedCOOFbr + usedCSLFbr));   
                HybX.fbrIdx[1].push_back(X.fbrIdx[1][fbr]);         
            }
             
            memcpy(&(HybX.inds[mode2][0]) + curIndSize, &(X.inds[mode2][0]) + idx_st, sizeof(ITYPE) * arSlcNnz[slc]);
            memcpy(&(HybX.vals[0]) + curIndSize, &(X.vals[0]) + idx_st, sizeof(DTYPE) * arSlcNnz[slc]);
            usedHCSRFbr += X.fbrPtr[0][slc + 1] - X.fbrPtr[0][slc];
            curIndSize += arSlcNnz[slc];
        }
    }
    
    HybX.fbrPtr[1].push_back(curIndSize);
    HybX.fbrPtr[0].push_back((ITYPE)(HybX.fbrPtr[1].size() -1 ));
    HybX.CSLslicePtr.push_back((ITYPE)(curCSLIndSize));

    HybX.nFibers = HybX.fbrPtr[1].size() - 1;
    HybX.COOnnz = COOindSize;//HybX.COOvals.size();
    HybX.CSLnnz = CSLindSize;
    HybX.HCSRnnz = indSize;//HybX.vals.size();
    if(Opt.verbose){
        cout << "slices in COO " <<HybX.COOnnz << endl;
        cout << "slices in CSL " <<HybX.CSLsliceIdx.size() << endl;
        cout << "slices in HCSR " <<HybX.fbrIdx[0].size() << endl;
    }
    return 0;
}
// TBD: diff with 3d..avoided CSL
inline int create_HYB_4D(HYBTensor &HybX, const Tensor &X, const Options &Opt){

    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    ITYPE mode3 = HybX.modeOrder[3];

    ITYPE fbrThreashold = Opt.fbrThreashold;

    ITYPE fiberNnz = 0;
    
    // reserving size 
    std::vector<int> arSlcNnz(X.fbrIdx[0].size(), 0);
    std::vector<bool> arFbrLenOne(X.fbrIdx[0].size(), true);
    std::vector<int> arFbrPtr(X.fbrPtr[1].size(), 0);

    int indSize = 0, CSLindSize = 0, COOindSize = 0, curIndSize = 0, curCSLIndSize = 0, curCOOIndSize = 0;

    // #pragma omp parallel 
    {
         // #pragma omp for private(fiberNnz) reduction(+:indSize, CSLindSize, COOindSize)
    
        for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
            
            for (int fbrS = X.fbrPtr[0][slc]; fbrS < X.fbrPtr[0][slc+1]; ++fbrS){   
            
                fiberNnz = 0;

                for (int fbr = X.fbrPtr[1][fbrS]; fbr < X.fbrPtr[1][fbrS+1]; ++fbr){   
                    
                    fiberNnz = X.fbrPtr[2][fbr+1] - X.fbrPtr[2][fbr];   
                    arSlcNnz[slc] += fiberNnz;
                    if(fiberNnz > 1) 
                        arFbrLenOne[slc] = false;  
                }
            }
            if( arSlcNnz[slc] == 1)
                COOindSize += arSlcNnz[slc];
            // else if(arFbrLenOne[slc])
            //     CSLindSize += arSlcNnz[slc];
            else 
                indSize += arSlcNnz[slc];
        }
    }

    for (int i = 0; i < X.ndims - 1; ++i){
        HybX.fbrPtr.push_back(std::vector<ITYPE>());
        HybX.fbrIdx.push_back(std::vector<ITYPE>());
    }

        // allocating COO space
    HybX.COOinds = (ITYPE **)malloc(sizeof(ITYPE *) * HybX.ndims);
    HybX.CSLinds = (ITYPE **)malloc(sizeof(ITYPE *) * HybX.ndims);
    HybX.inds = (ITYPE **)malloc(sizeof(ITYPE *) * HybX.ndims);

    for(int m = 0; m < HybX.ndims; m++)
    {
        HybX.COOinds[m] = (ITYPE (*))malloc(COOindSize * sizeof(ITYPE));
    }

    HybX.COOvals = (DTYPE*)malloc( COOindSize * sizeof(DTYPE));
    HybX.inds[mode3] = (ITYPE (*))malloc(indSize * sizeof(ITYPE));
    HybX.vals = (DTYPE*)malloc( indSize * sizeof(DTYPE));
    // enable for CSL
    // HybX.CSLinds[mode1] = (ITYPE (*))malloc(CSLindSize * sizeof(ITYPE));
    // HybX.CSLinds[mode2] = (ITYPE (*))malloc(CSLindSize * sizeof(ITYPE));
    // HybX.CSLvals = (DTYPE (*))malloc(CSLindSize * sizeof(DTYPE));

    // ITYPE sliceId, fiberId, sliceNnz = 0, fiberNnz = 0;
    // int usedCOOSlc = 0, usedCSLSlc = 0, usedHCSRSlc = 0;
    int usedCOOFbr = 0, usedCSLFbr = 0, usedHCSRFbr = 0;
      
    // for (int i = 0; i < X.ndims; ++i){
    //     HybX.COOinds.push_back(std::vector<ITYPE>()); 
    //     HybX.inds.push_back(std::vector<ITYPE>());
    //     HybX.CSLinds.push_back(std::vector<ITYPE>());
    //  }

    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

        const int stSFiber =  X.fbrPtr[0][slc];
        const int endSFiber =  X.fbrPtr[0][slc+1];
        const int stFiber =  X.fbrPtr[1][stSFiber];
        const int endFiber =  X.fbrPtr[1][endSFiber];
        const int idx_st = X.fbrPtr[2][stFiber];
        const int idx_end = X.fbrPtr[2][endFiber];
        
        if(arSlcNnz[slc] == 1){       

            HybX.COOinds[mode0][curCOOIndSize] = X.fbrIdx[0][slc];
            HybX.COOinds[mode1][curCOOIndSize] = X.fbrIdx[1][stSFiber];
            HybX.COOinds[mode2][curCOOIndSize] = X.fbrIdx[2][stFiber]; 
            HybX.COOinds[mode3][curCOOIndSize] = X.inds[mode3][idx_st];
            HybX.COOvals[curCOOIndSize] = X.vals[idx_st];  
            
            curCOOIndSize++; 
            usedCOOFbr++;      
        }
        // else if(fbrLenOne) {    
        //     HybX.CSLslicePtr.push_back(X.fbrPtr[0][slc] - (usedCOOFbr + usedHCSRFbr));
        //     HybX.CSLsliceIdx.push_back(X.fbrIdx[0][slc]);    
        //     for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){ 
        //         int idx =  X.fbrPtr[1][fbr];    
        //         HybX.CSLinds[mode1].push_back(X.fbrIdx[1][fbr]);
        //         HybX.CSLinds[mode2].push_back(X.inds[mode2][idx]); 
        //         HybX.CSLvals.push_back(X.vals[idx]);  
                
        //     }
        //     usedCSLFbr +=  X.fbrPtr[0][slc + 1] - X.fbrPtr[0][slc];
        //     usedCSLSlc++;
        // }
        else{

            HybX.fbrPtr[0].push_back(X.fbrPtr[0][slc] - (usedCOOFbr + usedCSLFbr));
            HybX.fbrIdx[0].push_back(X.fbrIdx[0][slc]);
            
            for (int fbrS = X.fbrPtr[0][slc]; fbrS < X.fbrPtr[0][slc+1]; ++fbrS){   
                
                HybX.fbrPtr[1].push_back(X.fbrPtr[1][fbrS] - (usedCOOFbr + usedCSLFbr));   
                HybX.fbrIdx[1].push_back(X.fbrIdx[1][fbrS]); 

                for (int fbr = X.fbrPtr[1][fbrS]; fbr < X.fbrPtr[1][fbrS+1]; ++fbr){   
                    HybX.fbrPtr[2].push_back(X.fbrPtr[2][fbr] - (usedCOOFbr + usedCSLFbr));   
                    HybX.fbrIdx[2].push_back(X.fbrIdx[2][fbr]); 
                    // copy(X.inds[mode3].begin() + X.fbrPtr[2][fbr] , X.inds[mode3].begin() + X.fbrPtr[2][fbr+1], std::back_inserter(HybX.inds[mode3]));       
                    // copy(X.vals.begin() + X.fbrPtr[2][fbr] , X.vals.begin() + X.fbrPtr[2][fbr+1], std::back_inserter(HybX.vals));     
                }     
            }
            memcpy(&(HybX.inds[mode3][0]) + curIndSize, &(X.inds[mode3][0]) + idx_st, sizeof(ITYPE) * arSlcNnz[slc]);
            memcpy(&(HybX.vals[0]) + curIndSize, &(X.vals[0]) + idx_st, sizeof(DTYPE) * arSlcNnz[slc]);

            usedHCSRFbr += X.fbrPtr[0][slc + 1] - X.fbrPtr[0][slc];
            curIndSize += arSlcNnz[slc];
        }
    }
    
    HybX.fbrPtr[2].push_back(indSize);
    HybX.fbrPtr[1].push_back((ITYPE)(HybX.fbrPtr[2].size() -1 ));
    HybX.fbrPtr[0].push_back((ITYPE)(HybX.fbrPtr[1].size() -1 ));
    // HybX.CSLslicePtr.push_back((ITYPE)(HybX.CSLvals.size()));

    HybX.nFibers = HybX.fbrPtr[2].size() - 1;
    HybX.COOnnz = COOindSize;//HybX.COOvals.size();
    HybX.CSLnnz = 0;;//HybX.CSLvals.size();
    HybX.HCSRnnz = indSize;//HybX.vals.size();

    if(Opt.verbose){
        cout << "slices in COO " <<HybX.COOnnz << endl;
        cout << "slices in CSL " <<HybX.CSLsliceIdx.size() << endl;
        cout << "slices in HCSR " <<HybX.fbrIdx[0].size() << endl;
    }
    return 0;
}

inline int create_TiledHCSR(TiledTensor *TiledX, const Options &Opt, int tile){

   ITYPE fbrThreashold = Opt.fbrThreashold;

    for (int i = 0; i < TiledX[tile].ndims - 1; ++i){
        TiledX[tile].fbrPtr.push_back(std::vector<ITYPE>());
        TiledX[tile].fbrIdx.push_back(std::vector<ITYPE>());
    }
    
    ITYPE mode0 = TiledX[tile].modeOrder[0];
    ITYPE mode1 = TiledX[tile].modeOrder[1];
    ITYPE mode2 = TiledX[tile].modeOrder[2];
    // ITYPE mode3 = TiledX[tile].modeOrder[3];

    std::vector<ITYPE> prevId(TiledX[tile].ndims-1);
    std::vector<ITYPE> fbrId(TiledX[tile].ndims-1);

    for (int i = 0; i < TiledX[tile].ndims-1; ++i){
        prevId[i] =  TiledX[tile].inds[TiledX[tile].modeOrder[i]][0];
        TiledX[tile].fbrPtr[i].push_back(0);
        TiledX[tile].fbrIdx[i].push_back(prevId[i]);
    }
    
    int idx = 1 ;
    
    while(idx < TiledX[tile].totNnz) {

        for (int i = 0; i < TiledX[tile].ndims-1; ++i) 
            fbrId[i] = TiledX[tile].inds[TiledX[tile].modeOrder[i]][idx];
   
        ITYPE fiberNnz = 1;
        bool sameFbr = true;

        for (int i = 0; i < TiledX[tile].ndims-1; ++i) {
            if(fbrId[i] != prevId[i])
                sameFbr = false;
        }
        /* creating last fiber consisting all nonzeroes in same fiber */
        while( sameFbr && idx < TiledX[tile].totNnz && fiberNnz < fbrThreashold){
            ++idx;
            fiberNnz++;
            for (int i = 0; i < TiledX[tile].ndims-1; ++i) {
                fbrId[i] = TiledX[tile].inds[TiledX[tile].modeOrder[i]][idx];   
                if(fbrId[i] != prevId[i])
                    sameFbr = false;
            }
        }

        if(idx == TiledX[tile].totNnz)
            break;

        /* TiledX[tile].ndims-2 is the last fiber ptr. Out of prev while loop means it is a new fiber. */
        TiledX[tile].fbrPtr[TiledX[tile].ndims-2].push_back(idx);
        TiledX[tile].fbrIdx[TiledX[tile].ndims-2].push_back(fbrId[TiledX[tile].ndims-2]);

        /* populating slice ptr and higher ptrs */
        for (int i = TiledX[tile].ndims - 3; i > -1 ; --i) {
            
            /* each dimension checks whether all parent/previous dimensions are in same slice/fiber/... */
            bool diffFbr = false;            
            int iDim = i;
            while(iDim > -1){
                if( fbrId[iDim] != prevId[iDim]) {//not else ..not become this in loop          
                    diffFbr = true;
                } 
                iDim--;
            }
            if(diffFbr){
                TiledX[tile].fbrIdx[i].push_back(fbrId[i]);
                TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size()) - 1);
            }
        }
     
        for (int i = 0; i < TiledX[tile].ndims-1; ++i)
            prevId[i] =  fbrId[i];

        ++idx;
        fiberNnz = 1;
    }
    TiledX[tile].fbrPtr[TiledX[tile].ndims-2].push_back(idx);
    TiledX[tile].fbrIdx[TiledX[tile].ndims-2].push_back(fbrId[TiledX[tile].ndims-2]);

    for (int i = TiledX[tile].ndims - 3; i > -1 ; --i)
        TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size() - 1 ));
    
    TiledX[tile].nFibers = TiledX[tile].fbrPtr[1].size() - 1;

    return 0;
}

// changed param to HYB
inline int make_HybBin(HYBTensor &X, const Options & Opt){

    ITYPE THREADLOAD = 2;
    ITYPE TB = 512;
    std::vector<ITYPE> UB;
    std::vector<ITYPE> LB;

    // Bin boundaries
    for (int i = 0; i < Opt.nBin; i++) {
        X.slcMapperBin.push_back(std::vector<ITYPE>());
        X.CSLslcMapperBin.push_back(std::vector<ITYPE>());
        UB.push_back((1 << i) * THREADLOAD + 1);
        LB.push_back(UB[i] >> 1);
    }

    LB[0] = 0;   UB[0] = 3;  // 1 WARP
    LB[1] = 2;   UB[1] = 5;  // 2 WARP
    LB[2] = 4;   UB[2] = 9;  // 4 WARP
    LB[3] = 8;   UB[3] = 17; // 8 WARP
    LB[4] = 16;   UB[4] = 1025;  // 16 WARP = 1 TB
    LB[5] = 1024;   UB[5] = 4 * TB + 1; // 32 WARP =2 TB
    LB[6] = 4 * TB;   UB[6] = 8 * TB + 1; // 64 WARP =4 TB
    LB[7] = 8 * TB;   UB[7] = 16 * TB + 1; // 128 WARP =8 TB
    LB[8] = 16 * TB;   UB[8] = 32 * TB + 1;  // 256 WARP = 16 TB
    LB[9] = 32 * TB ;   UB[9] = X.totNnz + 1;  // 512 WARP = 32 TB

    UB[Opt.nBin - 1] = X.totNnz + 1;
    UB[0] = 1025; // merging bins

    // Populate HCSR bin
    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

        int nnzSlc = 0;

        for (int fbrS = X.fbrPtr[0][slc]; fbrS < X.fbrPtr[0][slc+1]; ++fbrS){      
            
            if(X.ndims == 3)  
                nnzSlc += X.fbrPtr[1][fbrS+1] - X.fbrPtr[1][fbrS]; 
           
            else if(X.ndims == 4){
                for (int fbr = X.fbrPtr[1][fbrS]; fbr < X.fbrPtr[1][fbrS+1]; ++fbr){             
                    nnzSlc += X.fbrPtr[2][fbr+1] - X.fbrPtr[2][fbr]; 
                }
            }
        }
       
        // #pragma omp parallel
        // {
        // unsigned int cpu_thread_id = omp_get_thread_num();
        // int i = cpu_thread_id;
        for (int bin = 0; bin < Opt.nBin; ++bin){
            
            if (nnzSlc > LB[bin] && nnzSlc < UB[bin]) {
                X.slcMapperBin[bin].push_back(slc);
                break;
            }
        }
    }
    // // Populate CSL bin
    if(X.ndims == 3)  {
        for(ITYPE slc = 0; slc < X.CSLsliceIdx.size(); ++slc) {

            int fb_st = X.CSLslicePtr[slc];
            int fb_end = X.CSLslicePtr[slc+1];
            int nnzSlc = fb_end - fb_st; //nnz = nfibers
         
            // #pragma omp parallel
            // {
            // unsigned int cpu_thread_id = omp_get_thread_num();
            // int i = cpu_thread_id;
            for (int bin = 0; bin < Opt.nBin; ++bin)  {

                if (nnzSlc > LB[bin] && nnzSlc < UB[bin]) {
                    X.CSLslcMapperBin[bin].push_back(slc);
                    break;
                }
            }
        }
    }

    if(Opt.verbose){
        cout << "merged first 5 bins" << endl;
        for (int bin = 0; bin < Opt.nBin; ++bin)  
            cout << "CSL Bin "<<bin << ": " << X.CSLslcMapperBin[bin].size() << endl;
        for (int bin = 0; bin < Opt.nBin; ++bin)  
            cout << "HCSR Bin "<<bin << ": " << X.slcMapperBin[bin].size() << endl;
    }
    // debug
    // for (int i = 0; i < Opt.nBin; ++i)    {
    //     if(X.slcMapperBin[i].size() > 0){
    //         cout << "bin "<< i << ": "<< X.slcMapperBin[i].size() << endl;

    //         // for (int j = 0; j < X.slcMapperBin[i].size(); ++j)
    //         // {
    //         //     cout << X.sliceIdx[X.slcMapperBin[i][j]] << " ";
    //         // }
    //         cout << endl;
    //     }
    // }
}

inline int make_TiledBin(TiledTensor *TiledX, const Options & Opt, int tile){

    ITYPE THREADLOAD = 2;
    ITYPE TB = 512;
    std::vector<ITYPE> UB;
    std::vector<ITYPE> LB;

    // Bin boundaries
    for (int i = 0; i < Opt.nBin; i++) {
        TiledX[tile].slcMapperBin.push_back(std::vector<ITYPE>());
        UB.push_back((1 << i) * THREADLOAD + 1);
        LB.push_back(UB[i] >> 1);
    }

    LB[0] = 0;   UB[0] = 3;  // 1 WARP
    LB[1] = 2;   UB[1] = 5;  // 2 WARP
    LB[2] = 4;   UB[2] = 9;  // 4 WARP
    LB[3] = 8;   UB[3] = 17; // 8 WARP
    LB[4] = 16;   UB[4] = 1025;  // 16 WARP = 1 TB
    LB[5] = 1024;   UB[5] = 4 * TB + 1; // 32 WARP =2 TB
    LB[6] = 4 * TB;   UB[6] = 8 * TB + 1; // 64 WARP =4 TB
    LB[7] = 8 * TB;   UB[7] = 16 * TB + 1; // 128 WARP =8 TB
    LB[8] = 16 * TB;   UB[8] = 32 * TB + 1;  // 256 WARP = 16 TB
    LB[9] = 32 * TB ;   UB[9] = TiledX[tile].totNnz + 1;  // 512 WARP = 32 TB

    UB[Opt.nBin - 1] = TiledX[tile].totNnz + 1;
    if(Opt.verbose) 
        cout << "Merged all bins for smaller tiles" << endl;


    UB[0] = 1025; //mergin first 5 bin

    if(Opt.tileSize < 5000)
        UB[0] = 1025;
    // Populate bin
    for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {
        int nnzSlc = 0;

        for (int fbrS = TiledX[tile].fbrPtr[0][slc]; fbrS < TiledX[tile].fbrPtr[0][slc+1]; ++fbrS){      
            
            if(TiledX[tile].ndims == 3)  
                nnzSlc += TiledX[tile].fbrPtr[1][fbrS+1] - TiledX[tile].fbrPtr[1][fbrS]; 
           
            else if(TiledX[tile].ndims == 4){
                for (int fbr = TiledX[tile].fbrPtr[1][fbrS]; fbr < TiledX[tile].fbrPtr[1][fbrS+1]; ++fbr){             
                    nnzSlc += TiledX[tile].fbrPtr[2][fbr+1] - TiledX[tile].fbrPtr[2][fbr]; 
                }
            }
        }
        // #pragma omp parallel
        // {
        // unsigned int cpu_thread_id = omp_get_thread_num();
        // int i = cpu_thread_id;
        for (int bin = 0; bin < Opt.nBin; ++bin)
        {
            // cout << bin << " " << LB[bin] <<" " << UB[bin] << endl;
            if (nnzSlc > LB[bin] && nnzSlc < UB[bin]) {
                TiledX[tile].slcMapperBin[bin].push_back(slc);
                break;
            }
        }
    }

    if(Opt.verbose){
        for (int bin = 0; bin < Opt.nBin; ++bin)  
            cout << "Bin "<<bin << ": " << TiledX[tile].slcMapperBin[bin].size() << endl;
    }
}

inline int tensor_stats(const Tensor &X){

    ITYPE mode0 = X.modeOrder[0];
    int *nnzSlice = new int[X.fbrIdx[0].size()];
    int *nnzFibers = new int[X.nFibers];
    ITYPE totNnz = 0, flopsSaved = 0, emptySlc = 0;
    ITYPE minSlcNnz = 999999999, maxSlcNnz = 0;
    double stdDev = 0, stdDevFbr = 0;
    int avgSlcNnz = X.totNnz/X.dims[mode0]; // int to use stdDev
    int avgFbrNnz = X.totNnz/X.nFibers;

    // ofstream ofslc("slc_info.txt");
 
    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
        nnzSlice[slc] = 0;

        ITYPE idx0 = slc;
        int fb_st = X.fbrPtr[0][slc];
        int fb_end = X.fbrPtr[0][slc+1];
        // int nnzInFiber = fb_end - fb_st;
        
        for (int fbr = fb_st; fbr < fb_end; ++fbr){   
            int nnzInFiber = X.fbrPtr[1][fbr+1] - X.fbrPtr[1][fbr]; 
            nnzFibers[fbr] =  nnzInFiber;
            nnzSlice[slc] += nnzInFiber; 
            flopsSaved += nnzInFiber - 1;
            stdDevFbr += pow( avgFbrNnz - nnzFibers[fbr], 2);    
        }
        if(nnzSlice[slc] == 0) 
            emptySlc++;
        if (nnzSlice[slc] > maxSlcNnz) 
            maxSlcNnz = nnzSlice[slc];
        if (nnzSlice[slc] < minSlcNnz) 
            minSlcNnz = nnzSlice[slc];
        totNnz += nnzSlice[slc];
        stdDev += pow( avgSlcNnz - nnzSlice[slc], 2); 
        //ofslc << slc << " nFiber: " << X.fbrPtr[0][slc+1]- X.fbrPtr[0][slc] <<" nnzSlice: "<< nnzSlice[slc] 
        //<< " avg: "<< nnzSlice[slc] /(X.fbrPtr[0][slc+1]- X.fbrPtr[0][slc]) << endl;

    }
    cout << "flopsSaved " << " ,emptySlc " <<" ,minSlcNnz " << " ,maxSlcNnz " << " ,avgSlcNnz " << " ,stdDvSlcNnz "<< " ,stdDvFbrNnz " << ",nFibers ";
    cout << endl;
    
    cout << flopsSaved;
    cout << ", " << emptySlc <<", " << minSlcNnz << ", " << maxSlcNnz;
    cout << ", " << avgSlcNnz << ", "<< sqrt(stdDev/X.dims[mode0]) << ", "<< sqrt(stdDevFbr/X.nFibers);
    cout << ", " << X.nFibers << ", " ;//<< X.rwBin[0].size() << ", " << X.rwBin[1].size();
    cout << endl;

    if(totNnz == X.totNnz)
        cout << "nnz matched " << totNnz << endl;
    else
        cout << "nnz not matched! sliceNnz " << totNnz << " X.totNnz " << X.totNnz << endl;

    return 0;
}

/* param: MTX - mode wise tiled X */
inline int find_hvyslc_allMode(const Tensor &X, TiledTensor *MTX){
 
    int threshold = ( X.totNnz / X.dims[0] + X.totNnz / X.dims[1] + X.totNnz / X.dims[2]) / 3;

    /* initialize MICSF tiles */
    int mode = 0;

    for (int m = 0; m < X.ndims; ++m){
        bool switchBC = false;
        ITYPE switchMode;
        MTX[m].ndims = X.ndims;
        MTX[m].dims = new ITYPE[MTX[m].ndims];  

        //setting mode order accroding to mode length
        int  mMode1 = (1 + m) % X.ndims;
        int  mMode2 = (2 + m) % X.ndims;

        if( X.dims[mMode1] > X.dims[mMode2]) switchBC = true; else false;  
        
        for (int i = 0; i < X.ndims; ++i){
            MTX[m].inds.push_back(std::vector<ITYPE>());  
            MTX[m].dims[i] = X.dims[i];
            // MTX[m].modeOrder.push_back((i+m) % X.ndims);

            if(i > 0 && switchBC){

                if(i == 1) switchMode = 2;
                else if(i == 2) switchMode = 1;
            }
            else
                switchMode = i;       
            MTX[m].modeOrder.push_back((m + switchMode) % X.ndims);
        }         
    }
    for (int m = 0; m < X.ndims; ++m)

        cout << "mode order: " << m << ": " << MTX[m].modeOrder[0] << " " << MTX[m].modeOrder[1] << " "
        << MTX[m].modeOrder[2] << endl; 

    /* Populate with nnz for each slice for each mode */

    ITYPE mode0 = 0;//X.modeOrder[0];
    ITYPE mode1 = 1;//X.modeOrder[1];
    ITYPE mode2 = 2;//X.modeOrder[2];

    ITYPE *slcNnzMode0 = new ITYPE[X.dims[mode0]];
    ITYPE *slcNnzMode1 = new ITYPE[X.dims[mode1]];
    ITYPE *slcNnzMode2 = new ITYPE[X.dims[mode2]];

    memset(slcNnzMode0, 0, X.dims[mode0] * sizeof(ITYPE));
    memset(slcNnzMode1, 0, X.dims[mode1] * sizeof(ITYPE));
    memset(slcNnzMode2, 0, X.dims[mode2] * sizeof(ITYPE));
    
    for(ITYPE x=0; x<X.totNnz; ++x) {

        ITYPE idx0 = X.inds[mode0][x];
        ITYPE idx1 = X.inds[mode1][x];
        ITYPE idx2 = X.inds[mode2][x];
       
        slcNnzMode0[idx0]++;
        slcNnzMode1[idx1]++;
        slcNnzMode2[idx2]++;
    }

    for (int idx = 0; idx < X.totNnz; ++idx){

        ITYPE idx0 = X.inds[mode0][idx];
        ITYPE idx1 = X.inds[mode1][idx];
        ITYPE idx2 = X.inds[mode2][idx];

        if ( slcNnzMode2[idx2] > threshold )     
              mode = mode2;
        else if ( slcNnzMode1[idx1] > threshold )     
           mode = mode1;
        else
             mode = mode0;

        for (int i = 0; i < X.ndims; ++i)  {
            MTX[mode].inds[i].push_back(X.inds[i][idx]); 
        }
        MTX[mode].vals.push_back(X.vals[idx]);      
    }

    cout << "Threshold: "<< threshold << endl ;
     cout << "nnz in CSFs: ";
    for (int m = 0; m < X.ndims; ++m){
        MTX[m].totNnz = MTX[m].vals.size();
        cout << m << ": " << MTX[m].totNnz << " ";
    }
    cout << endl;
}

// inline int compute_accessK(Tensor &X, const Options &Opt){

//     ITYPE mode2 = X.modeOrder[2];
//     X.accessK = new ITYPE[X.dims[mode2]];
//     memset(X.accessK, 0, X.dims[mode2] * sizeof(ITYPE));
    
//     for(ITYPE x = 0; x < X.totNnz; ++x) {
    
//        ITYPE idx2 = X.inds[mode2][x];
//        X.accessK[idx2]++;
//     } 
//     // for (int i = 0; i <  X.dims[mode2]; ++i)
//     // {
//     //     cout << i <<": " << X.accessK[i] << endl; 
//     // }
// }

// inline int create_write_heavy(Tensor &X, const Options &Opt){

//     int shLimit = 192;
//     int nnzSlc = 0;
//     int nRwBin = 3;

//     for (int b = 0; b < nRwBin; ++b)
//     {
//         X.rwBin.push_back(std::vector<ITYPE>());
//     }
//     cout <<  X.rwBin[0].size() <<" " <<  X.rwBin[1].size() << endl;
//     for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {

//         for (int fbr = X.slicePtr[slc]; fbr < X.slicePtr[slc+1]; ++fbr){              
//             nnzSlc += X.fiberPtr[fbr+1] - X.fiberPtr[fbr]; 
//         }

//         //for now just write only bin
//         //bin 0 write heavy
//         //bin 1 ready heavy
//         //bin 3 equal == COO
//         if (nnzSlc > shLimit) {
//             X.rwBin[0].push_back(slc);
//         }
//         else
//             X.rwBin[1].push_back(slc);      
//     }
//     return 0;
// }


inline int prepare_Y(const Tensor &X, semiSpTensor &Y, const Options &Opt){
    Y.nRows = X.nFibers;
    Y.nCols = Opt.R;
    Y.vals = (DTYPE *)malloc( X.nFibers * Opt.R * sizeof(DTYPE));
    return 0;

}

inline int create_mats(const Tensor &X, Matrix *U, const Options &Opt, bool ata){
    
    ITYPE mode;
    ITYPE R = Opt.R;
    for (int m = 0; m < X.ndims; ++m){  
        mode = X.modeOrder[m];
        U[mode].nRows =  X.dims[mode];
        U[mode].nCols =  R;
        if(ata)  
            U[mode].nCols = U[mode].nRows;
        U[mode].vals = (DTYPE*)malloc(U[mode].nRows * U[mode].nCols * sizeof(DTYPE));
    }
    return 0;
}

inline int randomize_mats(const Tensor &X, Matrix *U, const Options &Opt){

    ITYPE mode;

    if(Opt.verbose)
        
        cout << endl << "rand func .9 " << endl; 
  
    for (int m = 0; m < X.ndims; ++m){  
        mode = X.modeOrder[m];
        srand48(0L);
        for(long r = 0; r < U[mode].nRows; ++r){
            for(long c = 0; c < U[mode].nCols; ++c) // or u[mode].nCols 
                U[mode].vals[r * U[mode].nCols + c] = 2;//0.1 * drand48(); //1 ;//(r * R + c + 1); //
        }
    }
    return 0;
}

inline int zero_mat(const Tensor &X, Matrix *U, ITYPE mode){

    srand48(0L);
    
    for(long r = 0; r < U[mode].nRows; ++r){
        for(long c = 0; c < U[mode].nCols; ++c) // or u[mode].nCols 
            U[mode].vals[r * U[mode].nCols +c] = 0;
    }
    return 0;
}

inline void write_output_ttmY(semiSpTensor &Y, ITYPE mode, string outFile){
    
    ofstream fp(outFile); 
    fp << Y.nRows << " x " << Y.nCols << " semiSpTensor" << endl;
    fp << std::fixed;
    for (int i = 0; i < Y.nRows; ++i)
    {
        for (int j = 0; j < Y.nCols; ++j)
        {
            fp << std::setprecision(2) << Y.vals[i * Y.nCols + j] << "\t" ;
        }
        fp << endl;  
    }
}


inline void print_matrix(Matrix *U, ITYPE mode){
    
    cout << U[mode].nRows << " x " << U[mode].nCols << " matrix" << endl;
    cout << std::fixed;
    for (int i = 0; i < U[mode].nRows; ++i)
    {
        for (int j = 0; j < U[mode].nCols; ++j)
        {
            cout << std::setprecision(2) << U[mode].vals[i * U[mode].nCols + j] << "\t" ;
        }
        cout << endl;  
    }
}

inline void write_output(Matrix *U, ITYPE mode, string outFile){
    
    ofstream fp(outFile); 
    fp << U[mode].nRows << " x " << U[mode].nCols << " matrix" << endl;
    fp << std::fixed;
    for (int i = 0; i < U[mode].nRows; ++i)
    {
        for (int j = 0; j < U[mode].nCols; ++j)
        {
            fp << std::setprecision(2) << U[mode].vals[i * U[mode].nCols + j] << "\t" ;
        }
        fp << endl;  
    }
}

inline void correctness_check(DTYPE *out, DTYPE *COOout, int nr, int nc){
   
    long mismatch = 0;
    DTYPE maxDiff = 0;
    DTYPE precision = 0.1;
    cout << std::fixed;
    for (int i = 0; i < nr; ++i){
        for (int j = 0; j < nc; ++j){
            DTYPE diff = abs(out[i * nc + j] - COOout[i * nc + j]);
            if( diff > precision){
                if(diff > maxDiff)
                    maxDiff = diff;
                // cout << "mismatch at (" << i <<"," << j <<") got: " << out[i * nc +j] << " exp: " << COOout[i * nc +j] << endl;
                mismatch++;
                // exit(0);
            }          
        }
    }

    if(mismatch == 0)
        cout << "Correctness pass!" << endl;
    else{
        cout <<  mismatch <<" mismatches found at " << precision << " precision" << endl;
        cout << "Maximum diff " << maxDiff << endl;
    }
}

inline void free_all(Tensor &X, semiSpTensor &Y, Matrix *U){
    
    free(Y.vals);

    for (int m = 0; m < X.ndims; ++m){  
        free(U[m].vals);
    }      

}

inline double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

inline void print_help_and_exit() {
    printf("options:\n\
        -R rank/feature : set the rank (default 32)\n\
        -m mode : set the mode of MTTKRP (default 0)\n\
        -t implementation type: 1: COO CPU, 2: HCSR CPU, 3: COO GPU 4: HCSR GPU (default 1)\n\
        -w warp per slice: set number of WARPs assign to per slice  (default 4)\n\
        -i output file name: e.g., ../dataset/delicious.tns \n\
        -o output file name: if not set not output file will be written\n");
       
    exit(1);
}

inline Options parse_cmd_options(int argc, char **argv) {
    
    Options param;
    int i;
    //handle options
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-')
            break;
        if (++i >= argc){
            print_help_and_exit();

        }
        
        switch (argv[i - 1][1]) {
        case 'R':
            param.R = atoi(argv[i]);
            break;
        case 'm':
            param.mode = atoi(argv[i]);
            break;

        case 't':
            param.impType = atoi(argv[i]);
            break;

        case 'w':
            param.warpPerSlice = atoi(argv[i]);
            break;

        case 'l':
            param.nTile = atoi(argv[i]);
            break;

        case 'f':
            param.fbrThreashold = atoi(argv[i]);
            break;

        case 'v':
            if(atoi(argv[i]) == 1)
                param.verbose = true;
            else
                param.verbose = false;
            break;

        case 'c':
            if(atoi(argv[i]) == 1)
                param.correctness = true;
            else
                param.correctness = false;
            break;

        case 'i':
            param.inFileName = argv[i];
            break;

        case 'o':
            param.outFileName = argv[i];
            break;

        default:
            fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
            print_help_and_exit();
            break;
        }
    }

  
    if (i > argc){
        cout << "weird " << argc << endl;
        print_help_and_exit();
    }

    return param;
}
#endif


