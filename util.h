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
        std::vector<vector<ITYPE>> inds;
        std::vector<DTYPE> vals;
        std::vector<vector<ITYPE>> fbrPtr;
        std::vector<vector<ITYPE>> fbrIdx;
        std::vector<vector<ITYPE>> slcMapperBin;
        std::vector<vector<ITYPE>> COOinds;
        std::vector<DTYPE> COOvals;
        std::vector<ITYPE> CSLslicePtr;
        std::vector<ITYPE> CSLsliceIdx;
        std::vector<vector<ITYPE>> CSLinds;
        std::vector<DTYPE> CSLvals;
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
    ITYPE mode = Opt.mode;

    ITYPE switchMode;

    ifstream fp(filename); 

    if(fp.fail()){
        cout << filename << " does not exist!" << endl;
        exit(0);
    }

    fp >> X.ndims; 

    X.dims = new ITYPE[X.ndims];

    for (int i = 0; i < X.ndims; ++i){
        // mode 0 never switches
        if(X.switchBC && i > 0){
            if(i == 1)
                switchMode = 2;
            else if(i == 2)
                switchMode = 1;
        }
        else
            switchMode = i;       
        X.modeOrder.push_back((switchMode + Opt.mode) % X.ndims);
    }

    for (int i = 0; i < X.ndims; ++i){
        // mode 0 never switches
        fp >> X.dims[i];      
        X.inds.push_back(std::vector<ITYPE>());
    }

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

    // for (int i = 0; i < X.totNnz; ++i)
    // {
    //  cout << X.inds[0][i] << " " << X.inds[1][i] << " "<< X.inds[2][i] <<endl;
    // }
    //    cout << "nnz " << X.totNnz << endl;
    return 0;
}

inline int print_COOtensor(const Tensor &X){

    for(ITYPE x = 0; x < X.totNnz; ++x) {
        for (int i = 0; i < X.ndims; ++i)
            cout << X.inds[X.modeOrder[i]][x] << " ";
        cout << endl;
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

    cout << "COO " << HybX.COOvals.size() << endl;

    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];

    for(ITYPE x = 0; x < HybX.COOvals.size(); ++x) {
    
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

    cout << "no of fibers " << TiledX[tile].fbrPtr[1].size() << endl;
    
    for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {

        ITYPE idx0 = TiledX[tile].fbrIdx[0][slc]; //slc
        int fb_st = TiledX[tile].fbrPtr[0][slc];
        int fb_end = TiledX[tile].fbrPtr[0][slc+1];
        // printf("slc st- end: %d %d %d \n", slc, fb_st, fb_end );
        
        for (int fbr = fb_st; fbr < fb_end; ++fbr){        
            // printf("fbr %d :  ", fbr );    
            for(ITYPE x = TiledX[tile].fbrPtr[1][fbr]; x < TiledX[tile].fbrPtr[1][fbr+1]; ++x) {
                cout << idx0 << " " << TiledX[tile].inds[1][x] << " " << TiledX[tile].inds[2][x] << endl;

            }            
        }
    }
}

inline int make_KTiling(const Tensor &X, TiledTensor *TiledX, const Options &Opt){

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE mode3 = ((X.ndims == 4) ? X.modeOrder[3] : 0) ;
    
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

    ITYPE fbrThreashold = Opt.fbrThreashold;

    for (int i = 0; i < X.ndims - 1; ++i){
        HybX.fbrPtr.push_back(std::vector<ITYPE>());
        HybX.fbrIdx.push_back(std::vector<ITYPE>());
    }

    bool fbrLenOne = true;

    ITYPE sliceId, fiberId, sliceNnz = 0, fiberNnz = 0;
    int usedCOOSlc = 0, usedCSLSlc = 0, usedHCSRSlc = 0;
    int usedCOOFbr = 0, usedCSLFbr = 0, usedHCSRFbr = 0;
    
    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    
    for (int i = 0; i < X.ndims; ++i){
        HybX.COOinds.push_back(std::vector<ITYPE>()); 
        HybX.inds.push_back(std::vector<ITYPE>());
        HybX.CSLinds.push_back(std::vector<ITYPE>());
     }

    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
        sliceNnz = 0;
        fbrLenOne = true;

        for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){  
            fiberNnz = X.fbrPtr[1][fbr+1] - X.fbrPtr[1][fbr]; 
            if(fiberNnz > 1) fbrLenOne = false;    
            sliceNnz += fiberNnz;
        }

        int stFiber =  X.fbrPtr[0][slc];
        int endFiber =  X.fbrPtr[0][slc+1];
        
        if(sliceNnz == 1){       
            int idx =  X.fbrPtr[1][X.fbrPtr[0][slc]];
            HybX.COOinds[mode0].push_back(X.fbrIdx[0][slc]);
            HybX.COOinds[mode1].push_back(X.fbrIdx[1][stFiber]);
            HybX.COOinds[mode2].push_back(X.inds[mode2][idx]); 
            HybX.COOvals.push_back(X.vals[idx]);  
            usedCOOSlc++;
            usedCOOFbr++;
        
        }
        else if(fbrLenOne) {    
            HybX.CSLslicePtr.push_back(X.fbrPtr[0][slc] - (usedCOOFbr + usedHCSRFbr));
            HybX.CSLsliceIdx.push_back(X.fbrIdx[0][slc]);    
            for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){ 
                int idx =  X.fbrPtr[1][fbr];    
                HybX.CSLinds[mode1].push_back(X.fbrIdx[1][fbr]);
                HybX.CSLinds[mode2].push_back(X.inds[mode2][idx]); 
                HybX.CSLvals.push_back(X.vals[idx]);  
                
            }
            usedCSLFbr +=  X.fbrPtr[0][slc + 1] - X.fbrPtr[0][slc];
            usedCSLSlc++;
        }
        else{

            HybX.fbrPtr[0].push_back(X.fbrPtr[0][slc] - (usedCOOFbr + usedCSLFbr));
            HybX.fbrIdx[0].push_back(X.fbrIdx[0][slc]);
            
            for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){   
                
                HybX.fbrPtr[1].push_back(X.fbrPtr[1][fbr] - (usedCOOFbr + usedCSLFbr));   
                HybX.fbrIdx[1].push_back(X.fbrIdx[1][fbr]); 
                copy(X.inds[mode2].begin() + X.fbrPtr[1][fbr] , X.inds[mode2].begin() + X.fbrPtr[1][fbr+1], std::back_inserter(HybX.inds[mode2]));       
                copy(X.vals.begin() + X.fbrPtr[1][fbr] , X.vals.begin() + X.fbrPtr[1][fbr+1], std::back_inserter(HybX.vals));          
            }
            usedHCSRFbr += X.fbrPtr[0][slc + 1] - X.fbrPtr[0][slc];
            usedHCSRSlc++;
        }
    }
    
    HybX.fbrPtr[1].push_back(HybX.inds[mode2].size());
    HybX.fbrPtr[0].push_back((ITYPE)(HybX.fbrPtr[1].size() -1 ));
    HybX.CSLslicePtr.push_back((ITYPE)(HybX.CSLvals.size()));
    HybX.nFibers = HybX.fbrPtr[1].size() - 1;
    HybX.COOnnz = HybX.COOvals.size();
    HybX.CSLnnz = HybX.CSLvals.size();
    HybX.HCSRnnz = HybX.vals.size();
    if(Opt.verbose){
        cout << "slices in COO " <<HybX.COOnnz << endl;
        cout << "slices in CSL " <<HybX.CSLsliceIdx.size() << endl;
        cout << "slices in HCSR " <<HybX.fbrIdx[0].size() << endl;
    }
    return 0;
}
// TBD: diff with 3d..avoided CSL
inline int create_HYB_4D(HYBTensor &HybX, const Tensor &X, const Options &Opt){

    ITYPE fbrThreashold = Opt.fbrThreashold;

    for (int i = 0; i < X.ndims - 1; ++i){
        HybX.fbrPtr.push_back(std::vector<ITYPE>());
        HybX.fbrIdx.push_back(std::vector<ITYPE>());
    }

    bool fbrLenOne = true;

    ITYPE sliceId, fiberId, sliceNnz = 0, fiberNnz = 0;
    int usedCOOSlc = 0, usedCSLSlc = 0, usedHCSRSlc = 0;
    int usedCOOFbr = 0, usedCSLFbr = 0, usedHCSRFbr = 0;
    
    ITYPE mode0 = HybX.modeOrder[0];
    ITYPE mode1 = HybX.modeOrder[1];
    ITYPE mode2 = HybX.modeOrder[2];
    ITYPE mode3 = HybX.modeOrder[3];
    
    for (int i = 0; i < X.ndims; ++i){
        HybX.COOinds.push_back(std::vector<ITYPE>()); 
        HybX.inds.push_back(std::vector<ITYPE>());
        HybX.CSLinds.push_back(std::vector<ITYPE>());
     }

    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
        sliceNnz = 0;
        fbrLenOne = true;


        for (int fbrS = X.fbrPtr[0][slc]; fbrS < X.fbrPtr[0][slc+1]; ++fbrS){   
            
            fiberNnz = 0;
            
            for (int fbr = X.fbrPtr[1][fbrS]; fbr < X.fbrPtr[1][fbrS+1]; ++fbr){   

                fiberNnz = X.fbrPtr[2][fbr+1] - X.fbrPtr[2][fbr]; 
                if(fiberNnz > 1) fbrLenOne = false; 
                sliceNnz += fiberNnz;
            }
        }

        int stFiber =  X.fbrPtr[0][slc];
        int endFiber =  X.fbrPtr[0][slc+1];
        
        if(sliceNnz == 1){       
            HybX.COOinds[mode0].push_back(X.fbrIdx[0][slc]);

            int fbrSIdx =  X.fbrPtr[0][slc];
            HybX.COOinds[mode1].push_back(X.fbrIdx[1][fbrSIdx]);

            int fbrIdx =  X.fbrPtr[1][fbrSIdx];
            HybX.COOinds[mode2].push_back(X.fbrIdx[2][fbrIdx]); 

            int idx =  X.fbrPtr[2][fbrIdx];
            HybX.COOinds[mode3].push_back(X.inds[mode3][idx]); 
            HybX.COOvals.push_back(X.vals[idx]);  
            
            usedCOOSlc++;
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
                    copy(X.inds[mode3].begin() + X.fbrPtr[2][fbr] , X.inds[mode3].begin() + X.fbrPtr[2][fbr+1], std::back_inserter(HybX.inds[mode3]));       
                    copy(X.vals.begin() + X.fbrPtr[2][fbr] , X.vals.begin() + X.fbrPtr[2][fbr+1], std::back_inserter(HybX.vals));     
                }     
            }
            usedHCSRFbr += X.fbrPtr[0][slc + 1] - X.fbrPtr[0][slc];
            usedHCSRSlc++;
        }
    }
    
    HybX.fbrPtr[2].push_back(HybX.inds[mode3].size());
    HybX.fbrPtr[1].push_back((ITYPE)(HybX.fbrPtr[2].size() -1 ));
    HybX.fbrPtr[0].push_back((ITYPE)(HybX.fbrPtr[1].size() -1 ));
    // HybX.CSLslicePtr.push_back((ITYPE)(HybX.CSLvals.size()));

    HybX.nFibers = HybX.fbrPtr[2].size() - 1;
    HybX.COOnnz = HybX.COOvals.size();
    HybX.CSLnnz = HybX.CSLvals.size();
    HybX.HCSRnnz = HybX.vals.size();

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

// inline int create_TiledHCSR(TiledTensor *TiledX, const Options &Opt , int tile){

//     ITYPE sliceId, fiberId;

//     for (int i = 0; i < TiledX[tile].ndims - 1; ++i){
//         TiledX[tile].fbrPtr.push_back(std::vector<ITYPE>());
//         TiledX[tile].fbrIdx.push_back(std::vector<ITYPE>());
//     }
//     ITYPE mode0 = TiledX[tile].modeOrder[0];
//     ITYPE mode1 = TiledX[tile].modeOrder[1];
//     ITYPE mode2 = TiledX[tile].modeOrder[2];
//     ITYPE fbrThreashold = Opt.fbrThreashold;
    
//     TiledX[tile].fbrPtr[0].push_back(0);
//     TiledX[tile].fbrPtr[1].push_back(0);
//     ITYPE prevSliceId =  TiledX[tile].inds[mode0][0];
//     ITYPE prevFiberId =  TiledX[tile].inds[mode1][0];
//     TiledX[tile].fbrIdx[0].push_back(prevSliceId);
//     TiledX[tile].fbrIdx[1].push_back(prevFiberId);
    
//     int idx = 1 ;
    
//     while(idx < TiledX[tile].totNnz) {
        
//         sliceId = TiledX[tile].inds[mode0][idx];
//         fiberId = TiledX[tile].inds[mode1][idx];   
   
//         ITYPE fiberNnz = 1;
//         while( fiberId == prevFiberId && sliceId == prevSliceId && idx < TiledX[tile].totNnz && fiberNnz < fbrThreashold){
//             ++idx;
//             fiberNnz++;
//             sliceId = TiledX[tile].inds[mode0][idx];
//             fiberId = TiledX[tile].inds[mode1][idx];           
//         }
//         if(idx == TiledX[tile].totNnz)
//             break;
//         TiledX[tile].fbrPtr[1].push_back(idx);
//         TiledX[tile].fbrIdx[1].push_back(fiberId);
        
//         if( sliceId != prevSliceId) {//not else ..not become this in loop
//             TiledX[tile].fbrIdx[0].push_back(sliceId);
//             TiledX[tile].fbrPtr[0].push_back((ITYPE)(TiledX[tile].fbrPtr[1].size()) - 1);
//         }      
//         prevSliceId = sliceId;
//         prevFiberId = fiberId;
//         ++idx;
//         fiberNnz = 1;
//     }
//     TiledX[tile].fbrPtr[1].push_back(idx);
//     TiledX[tile].fbrIdx[1].push_back(fiberId);
//     TiledX[tile].fbrPtr[0].push_back((ITYPE)(TiledX[tile].fbrPtr[1].size() -1 ));
//     TiledX[tile].nFibers = TiledX[tile].fbrPtr[1].size() - 1;

//     return 0;
// }
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
inline int create_mats(const Tensor &X, Matrix *U, const Options &Opt){
    
    ITYPE mode;
    ITYPE R = Opt.R;
    for (int m = 0; m < X.ndims; ++m){  
        mode = X.modeOrder[m];
        U[mode].nRows =  X.dims[mode];
        U[mode].nCols =  R;
        U[mode].vals = (DTYPE*)malloc(X.dims[mode] * R * sizeof(DTYPE));
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
                cout << "mismatch at (" << i <<"," << j <<") got: " << out[i * nc +j] << " exp: " << COOout[i * nc +j] << endl;
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


