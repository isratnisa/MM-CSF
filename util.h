#ifndef UTIL_H
#define UTIL_H

#define DTYPE float
#define ITYPE size_t // if chnage to unsigned int change the grid.x and gID in cuda kernel computation to long

#include <vector>
#include <algorithm>
#include <boost/sort/sort.hpp>
// #include <parallel/algorithm>
#include <iterator>
#include <unordered_map>
#include <map>
#include <boost/functional/hash.hpp>
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
        ITYPE *fbrLikeSlcInds;
        bool switchBC = false; // if true change matrix rand() to 1
        std::vector<ITYPE> modeOrder;
        std::vector<ITYPE> fbrCount;
        ITYPE **inds;
        DTYPE *vals;
        // std::vector<vector<ITYPE>> inds;
        // std::vector<DTYPE> vals;
        std::vector<vector<ITYPE>> fbrPtr;
        std::vector<vector<ITYPE>> fbrIdx;
        std::vector<vector<ITYPE>> slcMapperBin;
        ITYPE *nnzPerSlice;
        ITYPE *fiberPerSlice;
        ITYPE *nnzPerFiber;
        ITYPE *denseSlcPtr;
        ITYPE *partPerNnz;
        ITYPE *totnnzPerPart;
        unordered_map<pair<ITYPE, ITYPE>, ITYPE, boost::hash<pair<ITYPE, ITYPE>>> fbrHashTbl; 
        // map<pair<ITYPE, ITYPE>, ITYPE> fbrHashTbl;
};

class TiledTensor{
    public:
        ITYPE ndims;
        ITYPE *dims;
        ITYPE totNnz;
        ITYPE nFibers;
        ITYPE *accessK;
        ITYPE *fbrLikeSlcInds;
        std::vector<ITYPE> modeOrder;
        std::vector<ITYPE> fbrCount;
        ITYPE **inds;
        DTYPE *vals;
        // std::vector<vector<ITYPE>> inds;
        // std::vector<DTYPE> vals;
        std::vector<vector<ITYPE>> fbrPtr;
        std::vector<vector<ITYPE>> fbrIdx;
        std::vector<vector<ITYPE>> slcMapperBin;
        ITYPE *nnzPerSlice;
        ITYPE *fiberPerSlice;
        ITYPE *nnzPerFiber;
        ITYPE *denseSlcPtr;
        ITYPE *partPerNnz;
        ITYPE *totnnzPerPart;
        ITYPE *mpiEndSlc; 
        ITYPE *mpiEndFbr; 
        ITYPE *mpiEndNnz; 
        ITYPE *nnzInRank;
        ITYPE *fbrInRank;
        ITYPE *slcInRank;
        unordered_map<pair<ITYPE, ITYPE>, int, boost::hash<pair<ITYPE, ITYPE>>> fbrHashTbl; 
        // map<pair<ITYPE, ITYPE>, ITYPE> fbrHashTbl;
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
    ITYPE gridSize = 512;
    ITYPE TBsize = 128;
    ITYPE MIfbTh = 1;
    ITYPE fiberPerWarp = 1;
    bool verbose = false;     // if true change matrix rand() to 1
    bool correctness = false; 
    string inFileName; 
    string outFileName; 
    ITYPE nBin = 10;
    std::string m0 = "012";
    std::string m1 = "120";
    std::string m2 = "201";
    bool useMPI = true;
    bool natOrdering = false;
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

class MPI_param{
    public:
        int mpi_rank, n_proc;

};

inline int check_opt(const Tensor &X, Options &Opt){
    
    if(X.ndims > 4){
        cout << "Supported tensor dimension is 3 or 4." << endl;
        exit(0);
    }

    if(Opt.mode > X.ndims - 1){
        cout << "Mode cannot be larger than tensor dimension." << endl;
        exit(0);
    }

    if(Opt.impType == 11 || Opt.impType == 12 )
        Opt.mode = 0;

} 

inline int order_tensormode(Tensor &X, const Options &Opt, const int mode){

    int *sortMode = new int[X.ndims]; //sorted according to mode length
    int *natMode = new int[X.ndims]; // natural ordering
    bool *taken = new bool[X.ndims];
    int *sortModeLen = new int[X.ndims];

    for (int m = 0; m < X.ndims; ++m){
        natMode[m] = (m + mode) % X.ndims;
        sortModeLen[m] = X.dims[natMode[m]];
        taken[m] = false;
    }

    if(Opt.natOrdering){
        for (int i = 0; i < X.ndims; ++i)
            X.modeOrder.push_back(natMode[i]);
        cout << "Natural mode ordering " << endl;
    }
    else{
        /*linear sort of dimension length*/   
        int tmp, swapped;

        for (int i = 1; i < X.ndims; i++) {

            for (int j =i+1; j < X.ndims; j++) {
                
                if ( sortModeLen[i] > sortModeLen[j]) 
                    std::swap(sortModeLen[i],sortModeLen[j]);
            }
        }

        sortMode[0] = mode; 
        taken[mode] = true;

        for (int i = 1; i < X.ndims; i++) {

            for (int j = 0; j < X.ndims; j++) {
                
                if( sortModeLen[i] == X.dims[j] && !taken[j]){
                    sortMode[i] = j;
                    taken[j] = true;
                    break;
                }
            }
        }

        for (int i = 0; i < X.ndims; ++i)    
            X.modeOrder.push_back(sortMode[i]);
    }

    if(Opt.verbose){
        cout << "mode ordering: ";
        for (int i = 0; i < X.ndims; ++i)
            cout << X.modeOrder[i] << " ";
        cout << endl;
    }
}

inline int load_tensor(Tensor &X, const Options &Opt){
    
    if(Opt.verbose)
        cout << endl << "Loading tensor.." << endl;   
    
    string filename = Opt.inFileName;
    ITYPE index;
    DTYPE vid=0;

    ifstream fp(filename); 

    if(fp.fail()){
        cout << filename << " does not exist!" << endl;
        exit(0);
    }

    /*get number of line (totnnz)*/
    int numLines = 0;
    std::string unused;
    while ( std::getline(fp, unused) )
       ++numLines;
    X.totNnz = numLines - 2;

    fp.clear();                 // clear fail and eof bits
    fp.seekg(0, std::ios::beg);

    fp >> X.ndims; 

    X.dims = new ITYPE[X.ndims];

    for (int i = 0; i < X.ndims; ++i)
        fp >> X.dims[i]; 

    X.inds = new ITYPE*[X.ndims];  

    for(int i = 0; i < X.ndims; ++i)
        X.inds[i] = new ITYPE[X.totNnz];
    X.vals = new DTYPE[X.totNnz];   

    int idxCntr = 0;

    while(fp >> index) {
        
        X.inds[0][idxCntr] = index-1;
        for (int i = 1; i < X.ndims; ++i)
        {      
            fp >> index;
            X.inds[i][idxCntr] = index-1; 
        }
        fp >> vid;
        X.vals[idxCntr] = vid;
        idxCntr++;
    }

    order_tensormode(X, Opt, Opt.mode);

    return 0;
}

inline int init_tensor(Tensor *arrX, Tensor &X0, const Options &Opt, int mode){

    ITYPE switchMode = 0;
    bool switchBC =  false;
    arrX[mode].ndims = X0.ndims;
    arrX[mode].dims = new ITYPE[arrX[mode].ndims];
    arrX[mode].totNnz = X0.totNnz;

    arrX[mode].inds = X0.inds;
    arrX[mode].vals = X0.vals;
         
    for (int i = 0; i < arrX[mode].ndims; ++i)
        arrX[mode].dims[i] = X0.dims[i];

    order_tensormode(arrX[mode], Opt, mode);
}

inline bool sort_pred_3D(tuple <ITYPE, ITYPE, ITYPE, DTYPE> left, 
                  tuple <ITYPE, ITYPE, ITYPE, DTYPE> right) {

    if (get<0>(left) != get<0>(right)) 
        return (get<0>(left) < get<0>(right));
    
    return (get<1>(left) < get<1>(right));
    
}

inline bool sort_pred_4D(tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> left, 
                  tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> right) {
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
    ITYPE mode3;
    if(X.ndims == 4)
        mode3 = X.modeOrder[3];

    if(X.ndims == 3){

        vector < tuple <ITYPE, ITYPE, ITYPE, DTYPE> > items;
        tuple <ITYPE, ITYPE, ITYPE, DTYPE> ap;

        for (long idx = 0; idx < X.totNnz; ++idx) { 
    
            ap=std::make_tuple(X.inds[mode0][idx], X.inds[mode1][idx], X.inds[mode2][idx], X.vals[idx]);         
            items.push_back(ap);
        }
        // std::sort(std::parallel::par, items.begin(), items.end(), sort_pred);
        // std::sort(items.begin(), items.end(), sort_pred);
        boost::sort::sample_sort(items.begin(), items.end(), sort_pred_3D);

        for (long idx = 0; idx < X.totNnz; ++idx) {
            X.inds[mode0][idx] = get<0>(items[idx]);
            X.inds[mode1][idx] = get<1>(items[idx]);
            X.inds[mode2][idx] = get<2>(items[idx]);
            X.vals[idx] = get<3>(items[idx]);
        }
    }
    else if(X.ndims == 4){

        vector < tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> > items;
        tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> ap;

        for (long idx = 0; idx < X.totNnz; ++idx) { 
            
            ap=std::make_tuple(X.inds[mode0][idx], X.inds[mode1][idx], X.inds[mode2][idx], X.inds[mode3][idx], X.vals[idx]); 

            items.push_back(ap);
        }
        boost::sort::sample_sort(items.begin(), items.end(), sort_pred_4D);

        for (long idx = 0; idx < X.totNnz; ++idx) {

            X.inds[mode0][idx] = get<0>(items[idx]);
            X.inds[mode1][idx] = get<1>(items[idx]);
            X.inds[mode2][idx] = get<2>(items[idx]);           
            X.inds[mode3][idx] = get<3>(items[idx]);
            X.vals[idx] = get<4>(items[idx]);
        }
    }  
}

inline int sort_MI_CSF(const Tensor &X, TiledTensor *MTX, int m){

    const ITYPE mode0 = MTX[m].modeOrder[0];
    const ITYPE mode1 = MTX[m].modeOrder[1];
    const ITYPE mode2 = MTX[m].modeOrder[2];
    ITYPE mode3;
    if(X.ndims == 4)
        mode3 = MTX[m].modeOrder[3];

    if(X.ndims == 3){

        vector < tuple <ITYPE, ITYPE, ITYPE, DTYPE> > items;
        tuple <ITYPE, ITYPE, ITYPE, DTYPE> ap;
        
        for (long idx = 0; idx < MTX[m].totNnz; ++idx) { 

            ap=std::make_tuple(MTX[m].inds[mode0][idx], MTX[m].inds[mode1][idx], MTX[m].inds[mode2][idx], MTX[m].vals[idx]); 
            items.push_back(ap);
        }

        // sort(items.begin(), items.end(), sort_pred);
        boost::sort::sample_sort(items.begin(), items.end(), sort_pred_3D);
        
        #pragma omp parallel 
        {
        #pragma omp for 
        for (long idx = 0; idx < MTX[m].totNnz; ++idx) {

            MTX[m].inds[mode0][idx] = get<0>(items[idx]);
            MTX[m].inds[mode1][idx] = get<1>(items[idx]);
            MTX[m].inds[mode2][idx] = get<2>(items[idx]);
            MTX[m].vals[idx] = get<3>(items[idx]);
        }
        }
    }

    else if(X.ndims == 4){

        vector < tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> > items;
        tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> ap;

        for (long idx = 0; idx < MTX[m].totNnz; ++idx) { 
            
            ap=std::make_tuple(MTX[m].inds[mode0][idx], MTX[m].inds[mode1][idx], MTX[m].inds[mode2][idx], MTX[m].inds[mode3][idx], MTX[m].vals[idx]); 
            items.push_back(ap);
        }
        boost::sort::sample_sort(items.begin(), items.end(), sort_pred_4D);

        // #pragma omp parallel 
        {
        // #pragma omp for 
        for (long idx = 0; idx < MTX[m].totNnz; ++idx) {

            MTX[m].inds[mode0][idx] = get<0>(items[idx]);
            MTX[m].inds[mode1][idx] = get<1>(items[idx]);
            MTX[m].inds[mode2][idx] = get<2>(items[idx]);           
            MTX[m].inds[mode3][idx] = get<3>(items[idx]);
            MTX[m].vals[idx] = get<4>(items[idx]);
        }
        }
    } 

    // cout << "sorted tile : " << m << endl;
    // for (long idx = 0; idx < MTX[m].totNnz; ++idx) { 
    // std::cout << MTX[m].inds[0][idx] << " "
    //           << MTX[m].inds[1][idx] << " "
    //           << MTX[m].inds[2][idx] << " "
    //           << MTX[m].vals[idx] <<  std::endl;
    // }
}

inline int create_hashtable(Tensor &X){

    // X.fbrHashTbl.reserve(X.totNnz);

    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

        for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){      
           
            X.nnzPerFiber[fbr] = X.fbrPtr[1][fbr+1] - X.fbrPtr[1][fbr];
            X.fbrHashTbl[make_pair(X.fbrIdx[0][slc], X.fbrIdx[1][fbr])] = X.nnzPerFiber[fbr];
        }
    }
}

inline int print_COOtensor(const Tensor &X){

    cout << "Tensor X in COO format: " << endl;

    for(ITYPE x = 0; x < X.totNnz; ++x) {
        for (int i = 0; i < X.ndims; ++i)
            cout << X.inds[i][x] << " ";
        cout << X.vals[x]<< endl;
    }           
}

inline int print_TiledCOO(const TiledTensor *TiledX, const int tile){

    for(ITYPE x = 0; x < TiledX[tile].totNnz; ++x) {
        for (int i = 0; i < TiledX[tile].ndims; ++i)
            cout << TiledX[tile].inds[i][x] << " ";
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
                    cout << X.fbrIdx[0][slc] << " " << X.fbrIdx[1][fbr] << " " << X.inds[X.modeOrder[2]][x] << endl;
                if(mode0 == 1)
                    cout  << X.inds[X.modeOrder[2]][x] << " "<< X.fbrIdx[0][slc] <<" "<<X.fbrIdx[1][fbr] << " " <<endl;
                if(mode0 == 2)
                    cout  << X.fbrIdx[1][fbr]<<" "<< X.inds[X.modeOrder[2]][x]  << " "  << X.fbrIdx[0][slc] << endl;

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
                    
                    // if(mode0 == 0)
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
    
    if(TiledX[tile].ndims == 3){
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

    else if(TiledX[tile].ndims == 4){
        for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {
            
            for (int fbrS = TiledX[tile].fbrPtr[0][slc]; fbrS < TiledX[tile].fbrPtr[0][slc+1]; ++fbrS){   

                for (int fbr = TiledX[tile].fbrPtr[1][fbrS]; fbr < TiledX[tile].fbrPtr[1][fbrS+1]; ++fbr){        
         
                    for(ITYPE x = TiledX[tile].fbrPtr[2][fbr]; x < TiledX[tile].fbrPtr[2][fbr+1]; ++x) {
                        
                        // if(mode0 == 0)
                            cout << TiledX[tile].fbrLikeSlcInds[fbrS] << " " << TiledX[tile].fbrIdx[1][fbrS] << " " << TiledX[tile].fbrIdx[2][fbr] << " " << TiledX[tile].inds[3][x] << endl;
                        // if(mode0 == 1)
                        //     cout  << X.fbrIdx[1][fbr] << " " << X.inds[1][x] << " "<< X.fbrIdx[0][slc]; << endl;
                        // if(mode0 == 2)
                        //     cout  << X.inds[0][x]  << " " << X.fbrIdx[0][slc]; << " " << X.fbrIdx[1][fbr]<< endl;

                    }  
                }          
            }
        }
    }
}


inline int make_KTiling(const Tensor &X, TiledTensor *TiledX, Options &Opt){

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    ITYPE mode3 = ((X.ndims == 4) ? X.modeOrder[3] : 0) ;

    if(Opt.nTile > 1){
        cout << "Doesn't support tiling..fix kTiling as kTilingOld.. setting to 1";
        Opt.nTile = 1;
    }

    int tile = 0;
    
    //shortcut?? 
    TiledX[tile].totNnz = X.totNnz;

    // memcpy(&(HybX.CSLinds[mode1][0]) + curCSLIndSize, &(X.fbrIdx[1][0]) + stFiber, sizeof(ITYPE) * arSlcNnz[slc]);
           
    TiledX[tile].ndims = X.ndims;
    TiledX[tile].dims = new ITYPE[TiledX[tile].ndims]; 

    TiledX[tile].inds = new ITYPE*[X.ndims];

    for(int i = 0; i < X.ndims; ++i)
        TiledX[tile].inds[i] = new ITYPE[X.totNnz];
    
    TiledX[tile].vals = new DTYPE[X.totNnz];
    
    for (int i = 0; i < X.ndims; ++i){
        TiledX[tile].dims[i] = X.dims[i];
        TiledX[tile].modeOrder.push_back(X.modeOrder[i]);
    }           

    for (int idx = 0; idx < X.totNnz; ++idx){

        for (int i = 0; i < X.ndims; ++i)  
            TiledX[tile].inds[i][idx] = X.inds[i][idx]; 
        
        TiledX[tile].vals[idx] = X.vals[idx];    
    }  
}

// inline int make_KTiling(const Tensor &X, TiledTensor *TiledX, const Options &Opt){

//     ITYPE mode0 = X.modeOrder[0];
//     ITYPE mode1 = X.modeOrder[1];
//     ITYPE mode2 = X.modeOrder[2];
//     ITYPE mode3 = ((X.ndims == 4) ? X.modeOrder[3] : 0) ;
    
//     // cout << "TBD:: get rid of dims, mode, etc. for each tile";
//     for (int tile = 0; tile < Opt.nTile; ++tile){
//         TiledX[tile].ndims = X.ndims;
//         TiledX[tile].dims = new ITYPE[TiledX[tile].ndims]; 
        
//         for (int i = 0; i < X.ndims; ++i){
//             TiledX[tile].inds.push_back(std::vector<ITYPE>()); 
//             TiledX[tile].dims[i] = X.dims[i];
//             TiledX[tile].modeOrder.push_back(X.modeOrder[i]);
//         }           
//     }

//     int tile = 0;

//     for (int idx = 0; idx < X.totNnz; ++idx){

//         tile = ((TiledX[0].ndims == 3) ? X.inds[mode2][idx]/Opt.tileSize : X.inds[mode3][idx]/Opt.tileSize) ;

//         for (int i = 0; i < X.ndims; ++i)  {
//             TiledX[tile].inds[i].push_back(X.inds[i][idx]); 
//         }

//         TiledX[tile].vals.push_back(X.vals[idx]);      
//     }
//     for (int tile = 0; tile < Opt.nTile; ++tile){
//         TiledX[tile].totNnz = TiledX[tile].vals.size();
//     }

//     // Debug
//     // for (int tile = 0; tile < Opt.nTile; ++tile){
//     //     cout << "tile no: " << tile << endl;
        
//     //     for (int d = 0; d < TiledX[tile].vals.size(); ++d){
//     //         cout << TiledX[tile].inds[0][d] << " " << TiledX[tile].inds[1][d] 
//     //         <<" " << TiledX[tile].inds[2][d] ;
//     //         cout << endl;  
//     //     }      
//     //     cout << endl;     
//     // }
// }

inline int create_HCSR(Tensor &X, const Options &Opt){
 
    ITYPE fbrThreashold = Opt.fbrThreashold;
    if(Opt.impType == 12 )     
        fbrThreashold = 99999999;//

    for (int i = 0; i < X.ndims - 1; ++i){
        X.fbrPtr.push_back(std::vector<ITYPE>());
        X.fbrIdx.push_back(std::vector<ITYPE>());
    }

    std::vector<ITYPE> prevId(X.ndims-1);
    std::vector<ITYPE> fbrId(X.ndims-1);

    for (int i = 0; i < X.ndims-1; ++i){
        prevId[i] =  X.inds[X.modeOrder[i]][0];
        X.fbrPtr[i].push_back(0);
        X.fbrIdx[i].push_back(prevId[i]);
        X.fbrPtr[i].reserve(X.totNnz);
        X.fbrIdx[i].reserve(X.totNnz);
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
    
    X.nFibers = X.fbrPtr[1].size();

    // for (int i =0; i <  2 ;i++)
    //     X.inds[X.modeOrder[i]].resize(0);

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

    HybX.nFibers = HybX.fbrPtr[1].size();
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

    HybX.nFibers = HybX.fbrPtr[2].size();
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
    ITYPE fiberSNnz = 1;
    
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
            fiberSNnz++;
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
                /*splitting fbrS for 4D */
                else if( TiledX[tile].ndims == 4 && iDim == 1 && fiberSNnz > fbrThreashold){ 
                    diffFbr = true;                    
                }
                iDim--;
            }
            if(diffFbr){

                if(TiledX[tile].ndims == 4 && i == 1)
                    fiberSNnz = 1;

                TiledX[tile].fbrIdx[i].push_back(fbrId[i]);
                TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size()) - 1);
            }
        }
     
        for (int i = 0; i < TiledX[tile].ndims-1; ++i)
            prevId[i] =  fbrId[i];

        ++idx;
        fiberSNnz++;
        fiberNnz = 1;
        
    }
    TiledX[tile].fbrPtr[TiledX[tile].ndims-2].push_back(idx);
    TiledX[tile].fbrIdx[TiledX[tile].ndims-2].push_back(fbrId[TiledX[tile].ndims-2]);

    for (int i = TiledX[tile].ndims - 3; i > -1 ; --i)
        TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size() - 1 ));
    
    TiledX[tile].nFibers = TiledX[tile].fbrPtr[1].size();

    // cout << tile << " nnz: " <<  TiledX[tile].totNnz << " nslices: " <<  TiledX[tile].fbrPtr[0].size()  << " nFibers: " << TiledX[tile].nFibers << endl;
    // if(tile == TiledX[tile].ndims - 1){
    //     if(TiledX[tile].totNnz){
    //         int totslc = TiledX[0].fbrPtr[0].size() + TiledX[1].fbrPtr[0].size() +TiledX[2].fbrPtr[0].size();
    //         int totFbr = TiledX[0].fbrPtr[1].size() + TiledX[1].fbrPtr[1].size() +TiledX[2].fbrPtr[1].size();
    //         cout << "Total slice: " << totslc << " " << totFbr << endl;
    //     }
    // }

    return 0;
}


// inline int create_TiledHCSR(TiledTensor *TiledX, const Options &Opt, int tile){

//    ITYPE fbrThreashold = Opt.fbrThreashold;

//    TiledX[tile].fbrCount.resize(TiledX[tile].ndims);

//     // for (int i = 0; i < TiledX[tile].ndims - 1; ++i){
//     //     TiledX[tile].fbrPtr.push_back(std::vector<ITYPE>());
//     //     TiledX[tile].fbrIdx.push_back(std::vector<ITYPE>());
//     // }
    
//     ITYPE mode0 = TiledX[tile].modeOrder[0];
//     ITYPE mode1 = TiledX[tile].modeOrder[1];
//     ITYPE mode2 = TiledX[tile].modeOrder[2];
//     // ITYPE mode3 = TiledX[tile].modeOrder[3];

//     std::vector<ITYPE> prevId(TiledX[tile].ndims-1);
//     std::vector<ITYPE> fbrId(TiledX[tile].ndims-1);

//     // TiledX[tile].fbrPtr.resize( TiledX[tile].ndims, vector<ITYPE>(TiledX[tile].totNnz));
//     // TiledX[tile].fbrIdx.resize( TiledX[tile].ndims, vector<ITYPE>(TiledX[tile].totNnz));
    
//     for (int i = 0; i < TiledX[tile].ndims-1; ++i){
//         TiledX[tile].fbrCount[i] = 0;
//         prevId[i] =  TiledX[tile].inds[TiledX[tile].modeOrder[i]][0];
//         // TiledX[tile].fbrPtr[i].push_back(0);
//         // TiledX[tile].fbrIdx[i].push_back(prevId[i]);
//         TiledX[tile].fbrPtr[i][0]=0;
//         TiledX[tile].fbrIdx[i][0]=prevId[i];
//         TiledX[tile].fbrCount[i]++;
//     }
    
//     int idx = 1, curFb = 0;
    
//     while(idx < TiledX[tile].totNnz) {

//         for (int i = 0; i < TiledX[tile].ndims-1; ++i) 
//             fbrId[i] = TiledX[tile].inds[TiledX[tile].modeOrder[i]][idx];
   
//         ITYPE fiberNnz = 1;
//         bool sameFbr = true;

//         for (int i = 0; i < TiledX[tile].ndims-1; ++i) {
//             if(fbrId[i] != prevId[i])
//                 sameFbr = false;
//         }
//         /* creating last fiber consisting all nonzeroes in same fiber */
//         while( sameFbr && idx < TiledX[tile].totNnz && fiberNnz < fbrThreashold){
//             ++idx;
//             fiberNnz++;
//             for (int i = 0; i < TiledX[tile].ndims-1; ++i) {
//                 fbrId[i] = TiledX[tile].inds[TiledX[tile].modeOrder[i]][idx];   
//                 if(fbrId[i] != prevId[i])
//                     sameFbr = false;
//             }
//         }

//         if(idx == TiledX[tile].totNnz)
//             break;

//         /* TiledX[tile].ndims-2 is the last fiber ptr. Out of prev while loop means it is a new fiber. */
        
//         // TiledX[tile].fbrPtr[TiledX[tile].ndims-2].push_back(idx);
//         // TiledX[tile].fbrIdx[TiledX[tile].ndims-2].push_back(fbrId[TiledX[tile].ndims-2]);
//         curFb = TiledX[tile].fbrCount[TiledX[tile].ndims-2];
//         TiledX[tile].fbrPtr[TiledX[tile].ndims-2][curFb] = idx;
//         TiledX[tile].fbrIdx[TiledX[tile].ndims-2][curFb] = fbrId[TiledX[tile].ndims-2];
//         TiledX[tile].fbrCount[TiledX[tile].ndims-2]++;

//         /* populating slice ptr and higher ptrs */
//         for (int i = TiledX[tile].ndims - 3; i > -1 ; --i) {
            
//              // each dimension checks whether all parent/previous dimensions are in same slice/fiber/... 
//             bool diffFbr = false;            
//             int iDim = i;
//             while(iDim > -1){
//                 if( fbrId[iDim] != prevId[iDim]) {//not else ..not become this in loop          
//                     diffFbr = true;
//                 } 
//                 iDim--;
//             }
//             if(diffFbr){
//                 curFb = TiledX[tile].fbrCount[i];
//                 // TiledX[tile].fbrIdx[i].push_back(fbrId[i]);
//                 // TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size()) - 1);
//                 TiledX[tile].fbrIdx[i][curFb]=fbrId[i];
//                 TiledX[tile].fbrPtr[i][curFb]=((ITYPE)(TiledX[tile].fbrPtr[i+1].size()) - 1);
//                 TiledX[tile].fbrCount[i]++;
//             }
//         }
     
//         for (int i = 0; i < TiledX[tile].ndims-1; ++i)
//             prevId[i] =  fbrId[i];

//         ++idx;
//         fiberNnz = 1;
//     }
//     curFb = TiledX[tile].fbrCount[TiledX[tile].ndims-2];
//     // TiledX[tile].fbrPtr[TiledX[tile].ndims-2].push_back(idx);
//     // TiledX[tile].fbrIdx[TiledX[tile].ndims-2].push_back(fbrId[TiledX[tile].ndims-2]);
//     TiledX[tile].fbrPtr[TiledX[tile].ndims-2][curFb] = idx;
//     TiledX[tile].fbrIdx[TiledX[tile].ndims-2][curFb] = fbrId[TiledX[tile].ndims-2];
//     TiledX[tile].fbrCount[TiledX[tile].ndims-2]++;


//     for (int i = TiledX[tile].ndims - 3; i > -1 ; --i){
//         curFb = TiledX[tile].fbrCount[i];
//         // TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size() - 1 ));
//         TiledX[tile].fbrPtr[i][curFb] = (ITYPE)(TiledX[tile].fbrPtr[i+1].size() - 1 );
//     }

//     TiledX[tile].nFibers = TiledX[tile].fbrPtr[1].size();

//     // cout << tile << " nnz: " <<  TiledX[tile].totNnz << " nslices: " <<  TiledX[tile].fbrPtr[0].size()  << " nFibers: " << TiledX[tile].nFibers << endl;
//     // if(tile == TiledX[tile].ndims - 1){
//     //     if(TiledX[tile].totNnz){
//     //         int totslc = TiledX[0].fbrPtr[0].size() + TiledX[1].fbrPtr[0].size() +TiledX[2].fbrPtr[0].size();
//     //         int totFbr = TiledX[0].fbrPtr[1].size() + TiledX[1].fbrPtr[1].size() +TiledX[2].fbrPtr[1].size();
//     //         cout << "Total slice: " << totslc << " " << totFbr << endl;
//     //     }
//     // }

//     return 0;
// }

/** computes cold misses*/

inline int compute_reuse(Tensor &X, const Options &Opt){

    int mode2 = X.modeOrder[2];
    int mode1 = X.modeOrder[1];
    ITYPE *reuse = (ITYPE *)malloc( X.dims[mode2] * sizeof(ITYPE));
    ITYPE *reuseBin = (ITYPE *)malloc( Opt.nBin * sizeof(ITYPE));
    memset(reuse, 0, X.dims[mode2] * sizeof(ITYPE));    
    memset(reuseBin, 0, Opt.nBin * sizeof(ITYPE));  
    
    ITYPE *reuseFbr = (ITYPE *)malloc( X.dims[mode1]  * sizeof(ITYPE));
    memset(reuseFbr, 0, X.dims[mode1]  * sizeof(ITYPE));   
    long unqFbr = 0;
    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
        
        for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){ 

            if(reuseFbr[X.fbrIdx[1][fbr]] == 0)
                unqFbr++;

            reuseFbr[X.fbrIdx[1][fbr]]++; 

            for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) 

                reuse[X.inds[mode2][x]]++; 
        }
    }

    long usedFbr = 0;

    for (int i = 0; i < X.dims[mode1]; ++i) {
        
        if(reuseFbr[i] == 0)
            usedFbr++;
    }
    cout << usedFbr << " used fbr " << endl;

    std::vector<ITYPE> UB;
    std::vector<ITYPE> LB;

    // Bin boundaries
    for (int i = 0; i < Opt.nBin; i++) {
        UB.push_back(i);
        LB.push_back(i);
    }

    LB[0] = 0;   UB[0] = 2;  // 1 WARP
    LB[1] = 1;   UB[1] = 6;  // 2 WARP
    LB[2] = 6;   UB[2] = 11;  // 4 WARP
    LB[3] = 10;   UB[3] = 21; // 8 WARP
    LB[4] = 20;   UB[4] = 51;  // 16 WARP = 1 TB
    LB[5] = 50;   UB[5] = 101; // 32 WARP =2 TB
    LB[6] = 100;   UB[6] = 501; // 64 WARP =4 TB
    LB[7] = 500;   UB[7] = 1001; // 128 WARP =8 TB
    LB[8] = 1000;   UB[8] = 1501;  // 256 WARP = 16 TB
    LB[9] = 1500;   UB[9] = X.totNnz + 1;  // 512 WARP = 32 TB

    // /* re-use of C indices*/
    // for (int i = 0; i < X.dims[mode2]; ++i) {

    //     int curReuse =  reuse[i];

    //     for (int bin = 0; bin < Opt.nBin; ++bin){

    //         if (curReuse > LB[bin] && curReuse < UB[bin]) {
    //             reuseBin[bin]++;
    //             break;
    //         }
    //     }
    // }

    // if(Opt.verbose){
    //     for (int bin = 0; bin < Opt.nBin; ++bin)  
    //         cout << "reuse Bin "<<bin << ": " << reuseBin[bin] << endl;
    // }

    /* re-use of fibers*/

    memset(reuseBin, 0, Opt.nBin * sizeof(ITYPE));  
    for (int i = 0; i < X.dims[mode1]; ++i) {

        int curReuse =  reuseFbr[i];

        for (int bin = 0; bin < Opt.nBin; ++bin){

            if (curReuse > LB[bin] && curReuse < UB[bin]) {
                reuseBin[bin]++;
                break;
            }
        }
    }

    if(Opt.verbose){
        for (int bin = 0; bin < Opt.nBin; ++bin)  
            cout << "reuse Fiber Bin "<<bin << ": " << reuseBin[bin] << endl;
    }
    cout << "avg reuse " << unqFbr << " " << X.nFibers << endl;
}
inline int compute_reuse(TiledTensor *TiledX, const Options &Opt, int mode){

    int mode2 = TiledX[mode].modeOrder[2];
    int mode1 = TiledX[mode].modeOrder[1];
    ITYPE *reuse = (ITYPE *)malloc( TiledX[mode].dims[mode2] * sizeof(ITYPE));
    ITYPE *reuseBin = (ITYPE *)malloc( Opt.nBin * sizeof(ITYPE));
    memset(reuse, 0, TiledX[mode].dims[mode2] * sizeof(ITYPE));    
    memset(reuseBin, 0, Opt.nBin * sizeof(ITYPE));  
    
    ITYPE *reuseFbr = (ITYPE *)malloc( TiledX[mode].dims[mode1]  * sizeof(ITYPE));
    memset(reuseFbr, 0, TiledX[mode].dims[mode1]  * sizeof(ITYPE));   
    long unqFbr = 0;
    for(ITYPE slc = 0; slc < TiledX[mode].fbrIdx[0].size(); ++slc) {
        
        for (int fbr = TiledX[mode].fbrPtr[0][slc]; fbr < TiledX[mode].fbrPtr[0][slc+1]; ++fbr){ 

            if(reuseFbr[TiledX[mode].fbrIdx[1][fbr]] == 0)
                unqFbr++;
            reuseFbr[TiledX[mode].fbrIdx[1][fbr]]++; 


            for(ITYPE x = TiledX[mode].fbrPtr[1][fbr]; x < TiledX[mode].fbrPtr[1][fbr+1]; ++x) 

                reuse[TiledX[mode].inds[mode2][x]]++; 
        }
    }
    long usedFbr = 0;

    for(ITYPE slc = 0; slc < TiledX[mode].fbrIdx[0].size(); ++slc) {
        
        for (int fbr = TiledX[mode].fbrPtr[0][slc]; fbr < TiledX[mode].fbrPtr[0][slc+1]; ++fbr){ 

            if(reuseFbr[TiledX[mode].fbrIdx[1][fbr]] == 0)
                usedFbr++;
        }
    }
    cout << usedFbr << " used fbr " << endl;

    std::vector<ITYPE> UB;
    std::vector<ITYPE> LB;

    // Bin boundaries
    for (int i = 0; i < Opt.nBin; i++) {
        UB.push_back(i);
        LB.push_back(i);
    }

    LB[0] = 0;   UB[0] = 2;  // 1 WARP
    LB[1] = 1;   UB[1] = 6;  // 2 WARP
    LB[2] = 6;   UB[2] = 11;  // 4 WARP
    LB[3] = 10;   UB[3] = 21; // 8 WARP
    LB[4] = 20;   UB[4] = 51;  // 16 WARP = 1 TB
    LB[5] = 50;   UB[5] = 101; // 32 WARP =2 TB
    LB[6] = 100;   UB[6] = 501; // 64 WARP =4 TB
    LB[7] = 500;   UB[7] = 1001; // 128 WARP =8 TB
    LB[8] = 1000;   UB[8] = 1501;  // 256 WARP = 16 TB
    LB[9] = 1500;   UB[9] = TiledX[mode].totNnz + 1;  // 512 WARP = 32 TB

    // /* re-use of C indices*/
    // for (int i = 0; i < TiledX[mode].dims[mode2]; ++i) {

    //     int curReuse =  reuse[i];

    //     for (int bin = 0; bin < Opt.nBin; ++bin){

    //         if (curReuse > LB[bin] && curReuse < UB[bin]) {
    //             reuseBin[bin]++;
    //             break;
    //         }
    //     }
    // }

    // if(Opt.verbose){
    //     for (int bin = 0; bin < Opt.nBin; ++bin)  
    //         cout << "reuse Bin "<<bin << ": " << reuseBin[bin] << endl;
    // }

    /* re-use of fibers*/

    memset(reuseBin, 0, Opt.nBin * sizeof(ITYPE));  
    for (int i = 0; i < TiledX[mode].dims[mode1]; ++i) {

        int curReuse =  reuseFbr[i];

        for (int bin = 0; bin < Opt.nBin; ++bin){

            if (curReuse > LB[bin] && curReuse < UB[bin]) {
                reuseBin[bin]++;
                break;
            }
        }
    }

    if(Opt.verbose){
        for (int bin = 0; bin < Opt.nBin; ++bin)  
            cout << "reuse Fiber Bin "<<bin << ": " << reuseBin[bin] << endl;
    }
    cout << "avg reuse " << usedFbr << " " << unqFbr <<" " << TiledX[mode].nFibers << endl;
}

inline int compute_reuse_distance(Tensor &X, const Options &Opt){

    int mode1 = X.modeOrder[1];
    int mode2 = X.modeOrder[2];
    int inf = X.totNnz;
    //must be int
    int *lastLocNnz = (int *)malloc( X.dims[mode2] * sizeof(int));
    ITYPE *distCntrNnz = (ITYPE *)malloc( X.totNnz * sizeof(ITYPE));

    int *lastLocFbr = (int *)malloc( X.dims[mode1] * sizeof(int));
    ITYPE *distCntrFbr = (ITYPE *)malloc( X.totNnz * sizeof(ITYPE));

    for (int i = 0; i < X.dims[mode2]; ++i)
        lastLocNnz[i] = -1;
    
    for (int i = 0; i < X.dims[mode1]; ++i)
        lastLocFbr[i] = -1;
       
    memset(distCntrNnz, 0, X.totNnz * sizeof(ITYPE));      
    memset(distCntrFbr, 0, X.totNnz * sizeof(ITYPE));   
    
    ITYPE *distBin = (ITYPE *)malloc( Opt.nBin * sizeof(ITYPE));
    memset(distBin, 0, Opt.nBin * sizeof(ITYPE));    
    
    int distNnz = 0, prevLocNnz, distFbr = 0, prevLocFbr;
    
    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
        
        for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){ 

            /* collect reuse distance from fiber indices*/
            prevLocFbr = lastLocFbr[X.fbrIdx[1][fbr]];            
            if(prevLocFbr == -1)
                distFbr = inf;
            else
                distFbr = fbr - prevLocFbr;
            distCntrFbr[distFbr]++;
            lastLocFbr[X.fbrIdx[1][fbr]] = fbr;

            for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {
                
                /* collect reuse distance from nnz indices*/
                prevLocNnz = lastLocNnz[X.inds[mode2][x]];
                if(prevLocNnz == -1)
                    distNnz = inf;
                else
                    distNnz = x - prevLocNnz;
                distCntrNnz[distNnz]++;
                lastLocNnz[X.inds[mode2][x]] = x;
            }   
        }
    }

    std::vector<ITYPE> UB;
    std::vector<ITYPE> LB;

    // Bin boundaries
    for (int i = 0; i < Opt.nBin; i++) {
        UB.push_back(i);
        LB.push_back(i);
    }

    LB[0] = 0;   UB[0] = 2;  // 1 WARP
    LB[1] = 1;   UB[1] = 6;  // 2 WARP
    LB[2] = 6;   UB[2] = 11;  // 4 WARP
    LB[3] = 10;   UB[3] = 21; // 8 WARP
    LB[4] = 20;   UB[4] = 51;  // 16 WARP = 1 TB
    LB[5] = 50;   UB[5] = 101; // 32 WARP =2 TB
    LB[6] = 100;   UB[6] = 501; // 64 WARP =4 TB
    LB[7] = 500;   UB[7] = 1001; // 128 WARP =8 TB
    LB[8] = 1000;   UB[8] = 1501;  // 256 WARP = 16 TB
    LB[9] = 1500;   UB[9] = X.totNnz + 1;  // 512 WARP = 32 TB

    /* re-use of C indices*/
    for (int i = 0; i < X.totNnz; ++i) {

        int curDist =  distCntrNnz[i];

        for (int bin = 0; bin < Opt.nBin; ++bin){

            if (curDist > LB[bin] && curDist < UB[bin]) {
                distBin[bin]++;
                break;
            }
        }
    }

    // if(Opt.verbose)
    {
        for (int bin = 0; bin < Opt.nBin; ++bin)  
            cout << "nnz Bin "<<bin << ": " << distBin[bin] << endl;
    }

    /* re-use of fibers*/

    memset(distBin, 0, Opt.nBin * sizeof(ITYPE));  
    
    for (int i = 0; i < X.totNnz; ++i) {

        int curDist =  distCntrFbr[i];

        for (int bin = 0; bin < Opt.nBin; ++bin){

            if (curDist > LB[bin] && curDist < UB[bin]) {
                distBin[bin]++;
                break;
            }
        }
    }

    // if(Opt.verbose)
    {
        for (int bin = 0; bin < Opt.nBin; ++bin)  
            cout << "reuse Fiber Bin "<<bin << ": " << distBin[bin] << endl;
    }
}


inline int create_fbrLikeSlcInds(Tensor &X, const Options &Opt){

    X.fbrLikeSlcInds = (ITYPE *)malloc( X.nFibers * sizeof(ITYPE));
    
    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {
        
        for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){  

            X.fbrLikeSlcInds[fbr] =   X.fbrIdx[0][slc] ;     
        }
    }
}

inline int create_fbrLikeSlcInds(TiledTensor *TiledX, int mode){

    TiledX[mode].fbrLikeSlcInds = (ITYPE *)malloc( TiledX[mode].nFibers * sizeof(ITYPE));

    #pragma omp parallel 
    {
    #pragma omp for    
    for(ITYPE slc = 0; slc < TiledX[mode].fbrIdx[0].size(); ++slc) {
        
        for (int fbr = TiledX[mode].fbrPtr[0][slc]; fbr < TiledX[mode].fbrPtr[0][slc+1]; ++fbr){  

            TiledX[mode].fbrLikeSlcInds[fbr] =   TiledX[mode].fbrIdx[0][slc] ;     
        }
    }
    }
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
        cout << "Merged all bins for smaller tiles, added nFiber as bin info" << endl;

    UB[0] = 1025; //mergin first 5 bin

    // Populate bin
    for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {
        int nnzSlc = 0;
        nnzSlc += TiledX[tile].fbrPtr[0][slc+1] - TiledX[tile].fbrPtr[0][slc];

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

    // if(Opt.verbose){
    //     for (int bin = 0; bin < Opt.nBin; ++bin)  
    //         cout << "Bin "<<bin << ": " << TiledX[tile].slcMapperBin[bin].size() << endl;
    // }
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
    int singleSliceFromAllMode = 0;

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
    cout << ", " << (X.dims[mode0] - X.fbrIdx[0].size()) <<", " << minSlcNnz << ", " << maxSlcNnz;
    cout << ", " << avgSlcNnz << ", "<< sqrt(stdDev/X.dims[mode0]) << ", "<< sqrt(stdDevFbr/X.nFibers);
    cout << ", " << X.nFibers << ", " ;//<< X.rwBin[0].size() << ", " << X.rwBin[1].size();
    cout << endl;

    if(totNnz == X.totNnz)
        cout << "nnz matched " << totNnz << endl;
    else
        cout << "nnz not matched! sliceNnz " << totNnz << " X.totNnz " << X.totNnz << endl;

    return 0;
}

inline int get_nnzPerFiberData(Tensor &X){

    int sliceMode=X.modeOrder[0];
    int fiberMode=X.modeOrder[1];

    X.nnzPerFiber = new ITYPE[X.nFibers];
    memset(X.nnzPerFiber, 0, X.nFibers * sizeof(ITYPE));     

    X.nnzPerSlice = new ITYPE[X.dims[sliceMode]];
    memset(X.nnzPerSlice, 0, X.dims[sliceMode] * sizeof(ITYPE)); 

    cout << "do it parallel" << endl; 

    for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

        for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){      
           
            X.nnzPerFiber[fbr] = X.fbrPtr[1][fbr+1] - X.fbrPtr[1][fbr];
            X.nnzPerSlice[X.fbrIdx[0][slc]] += X.nnzPerFiber[fbr];
        }
    }
    return 0;
}

// inline int populate_paritions(const Tensor &X, TiledTensor *MTX){
 
//     // avoid pushback by using tot nnzperpart
//     // do parallel

//     for (int idx = 0; idx < X.totNnz; ++idx){
//         int mode = X.partPerNnz[idx];
//         for (int i = 0; i < X.ndims; ++i)  {
//             MTX[mode].inds[i].push_back(X.inds[i][idx]); 
//         }
//         MTX[mode].vals.push_back(X.vals[idx]);    
//     }
//     for (int i = 0; i < X.ndims; ++i)
//             MTX[i].totNnz = MTX[i].vals.size();
//     return 0;
// }

inline int populate_paritions(Tensor &X, TiledTensor *MTX){
 
    // avoid pushback by using tot nnzperpart
    ITYPE *nnzCntr = new ITYPE[X.ndims];
    memset(nnzCntr, 0, X.ndims * sizeof(ITYPE));  
    
    int mode;
    for (int idx = 0; idx < X.totNnz; ++idx){
        mode = X.partPerNnz[idx];
        X.totnnzPerPart[mode]++;
    }

    for(int i = 0; i < X.ndims; ++i){
        
        MTX[i].inds = new ITYPE*[X.ndims];
        MTX[i].totNnz = X.totnnzPerPart[i];
        
        for(int m = 0; m < X.ndims; ++m){
            
            MTX[i].inds[m] = new ITYPE[X.totnnzPerPart[i]];
        }
        MTX[i].vals = new DTYPE[X.totnnzPerPart[i]];
    }
    
    // do parallel
    // #pragma omp parallel for reduction(+:nnzCntr[mode])
    for (int idx = 0; idx < X.totNnz; ++idx){
        
        int mode = X.partPerNnz[idx];
        
        for (int i = 0; i < X.ndims; ++i)  {
            MTX[mode].inds[i][nnzCntr[mode]] = X.inds[i][idx]; 
        }
        MTX[mode].vals[nnzCntr[mode]] = X.vals[idx];    
        nnzCntr[mode]++;
    }

    return 0;
}

inline int binarySearch1(ITYPE *arr, ITYPE left, ITYPE right, ITYPE value) { 
  
    while (left <= right) {
            // int middle = (left + right) / 2;
            int middle = ((unsigned int)left+(unsigned int)right) >> 1;
            if (arr[middle] == value)
                  return middle;
            else if (arr[middle] > value)
                  right = middle - 1;
            else
                  left = middle + 1;
      }
      return -1;
}

inline int binarySearch(ITYPE *arr, ITYPE l, ITYPE r, ITYPE x) { 
    
    if (r >= l) { 
        // int mid = ((unsigned int)left+(unsigned int)right) >> 1;
        // ITYPE mid = l + (r - l) / 2; 
        unsigned int mid = ((unsigned int)l + (unsigned int)r) >> 1;

        if (arr[mid] == x) 
            return mid; 
  
        if (arr[mid] > x) 
            return binarySearch(arr, l, mid - 1, x); 

        return binarySearch(arr, mid + 1, r, x); 
    } 
    return -1; 
}

inline int maxOf3( int a, int b, int c )
{
   int max = ( a < b ) ? b : a;
   return ( ( max < c ) ? c : max );
}

inline int mm_partition_reuseBased(Tensor *arrX, Tensor &X, TiledTensor *MTX, Options & Opt){

    X.partPerNnz = new ITYPE[X.totNnz];
    memset(X.partPerNnz, 0, X.totNnz * sizeof(ITYPE));  
    X.totnnzPerPart = new ITYPE[X.ndims];
    memset(X.totnnzPerPart, 0, X.ndims * sizeof(ITYPE));  
 
    for (int m = 0; m < arrX[0].ndims; ++m){
        
        int sliceMode=arrX[m].modeOrder[0];
        int fiberMode=arrX[m].modeOrder[1];
        
        arrX[m].nnzPerFiber = new ITYPE[arrX[m].nFibers];
        memset(arrX[m].nnzPerFiber, 0, arrX[m].nFibers * sizeof(ITYPE));     

        arrX[m].nnzPerSlice = new ITYPE[arrX[m].dims[sliceMode]];
        memset(arrX[m].nnzPerSlice, 0, arrX[m].dims[sliceMode] * sizeof(ITYPE));  

        arrX[m].denseSlcPtr = (ITYPE*)malloc( (arrX[m].dims[sliceMode]+1) * sizeof(ITYPE)); //new ITYPE[arrX[m].dims[sliceMode]];
        memset(arrX[m].denseSlcPtr, 0, (arrX[m].dims[sliceMode] + 1 ) * sizeof(ITYPE));  
    }

    /*creating dense slices so that nnz can directly index slices unlike fiber. For
    fiber it needs to scan all fibers in a slice. */

    for (int m = 0; m < arrX[0].ndims; ++m){

        #pragma omp parallel 
        {
        #pragma omp for 
        for(ITYPE slc = 0; slc < arrX[m].fbrIdx[0].size(); ++slc) {

            arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]] = arrX[m].fbrPtr[0][slc];
                   
            if(slc == arrX[m].fbrIdx[0].size()-1)
                arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]+1] = arrX[m].fbrPtr[0][slc];
            else
                arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]+1] = arrX[m].fbrPtr[0][slc+1];

            /* Populate nnz per fiber and nnz per slice */
            for (int fbr = arrX[m].fbrPtr[0][slc]; fbr < arrX[m].fbrPtr[0][slc+1]; ++fbr){      
                
                if(X.ndims == 3){   
                    arrX[m].nnzPerFiber[fbr] = arrX[m].fbrPtr[1][fbr+1] - arrX[m].fbrPtr[1][fbr];
                    arrX[m].nnzPerSlice[arrX[m].fbrIdx[0][slc]] += arrX[m].nnzPerFiber[fbr];
                }

                else if(X.ndims == 4){  

                    for (int fbrIn = arrX[m].fbrPtr[1][fbr]; fbrIn < arrX[m].fbrPtr[1][fbr+1]; ++fbrIn)          
                        arrX[m].nnzPerFiber[fbr] += arrX[m].fbrPtr[2][fbrIn+1] - arrX[m].fbrPtr[2][fbrIn];
            
                    arrX[m].nnzPerSlice[arrX[m].fbrIdx[0][slc]] += arrX[m].nnzPerFiber[fbr];
                }
            }
        }
        }
    }

    // #pragma omp barrier

    int threshold = ( X.totNnz / X.dims[0] + X.totNnz / X.dims[1] + X.totNnz / X.dims[2]) / 3;
    int thNnzInTile = X.totNnz*1;

    /* initialize MICSF tiles */
    int mode = 0;

    for (int m = 0; m < X.ndims; ++m){

        MTX[m].ndims = X.ndims;
        MTX[m].dims = new ITYPE[MTX[m].ndims];  
        MTX[m].totNnz = 0; // WHY mode?
        
        for (int i = 0; i < X.ndims; ++i){
            MTX[m].modeOrder.push_back(arrX[m].modeOrder[i]); 
            MTX[m].dims[i] = X.dims[i];
        }     
    }    

    /* Populate with nnz for each slice for each mode */

    ITYPE mode0 = 0;//std::min(X.dims[0], X.dims[1], X.dims[2]);
    ITYPE mode1 = 1;//X.modeOrder[1];
    ITYPE mode2 = 2;//X.modeOrder[2];
    ITYPE mode3 = 3;

    //not mode sorted
    int shortestMode = ( (X.dims[X.modeOrder[0]] <= X.dims[X.modeOrder[1]]) ? X.modeOrder[0] : X.modeOrder[1]) ;

    bool sameFm0m1 = false, sameFm0m2 = false, sameFm1m2 = false, sameFm0m3 = false, 
        sameFm1m3 = false, sameFm2m3 = false;

    int fbTh =  Opt.MIfbTh;
    int slTh =  1, shortMode = 0;

    for (int m = 0; m < X.ndims; ++m){

        if(m == 1){
            if (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
                sameFm0m1 = true;
                shortMode = (arrX[m].dims[0] <= arrX[m].dims[1] ? 0 : 1);
            }
        }
        else if(m == 2){
            if(arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]){
                sameFm0m2 = true;
                shortMode = (arrX[m].dims[0] <= arrX[m].dims[2] ? 0 : 2);
            }
            else if ( arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
                sameFm1m2 = true;
                shortMode = (arrX[m].dims[1] <= arrX[m].dims[2] ? 1 : 2);
            }
        }
        else if(m == 3){
            if(arrX[m].modeOrder[1] == arrX[m-3].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-3].modeOrder[1]){
                sameFm0m3 = true;
                shortMode = (arrX[m].dims[0] <= arrX[m].dims[3] ? 0 : 3);
            }
            else if ( arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]){
                sameFm1m3 = true;
                shortMode = (arrX[m].dims[1] <= arrX[m].dims[3] ? 1 : 3);
            }
            else if ( arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
                sameFm2m3 = true;
                shortMode = (arrX[m].dims[2] <= arrX[m].dims[3] ? 2 : 3);
            }
        }
    }
    bool casePr = false;
    int allnnz1 = 0;
    #pragma omp barrier

    ITYPE **slcNnzPerParti = new ITYPE*[X.ndims];
    
    for (int m = 0; m < X.ndims; ++m){
    
        slcNnzPerParti[m] = new ITYPE[X.dims[m]];
        memset(&(slcNnzPerParti[m][0]), 0, X.dims[m] * sizeof(ITYPE));
    }  

    /******** Process NNZ********s*/
    // #pragma omp parallel 
    {
    ITYPE *fbrNnz = new ITYPE[X.ndims];
    ITYPE *fbrNo = new ITYPE[X.ndims];
    ITYPE *curIdx = new ITYPE[X.ndims];
    ITYPE *sliceNnz =  new ITYPE[X.ndims];
    ITYPE tmpSlc;
    int mode;

    // #pragma omp for 
    for (int idx = 0; idx < X.totNnz; ++idx){
        
        bool modeDone = false;
        bool allsame = true;

        // if(idx%1000000 == 0 && idx < 10000000 )  casePr = true;
        // else casePr = false;
   
        for (int m = 0; m < X.ndims; ++m)
            curIdx[m] = X.inds[m][idx];

        /*Finding fiber nnz*/
        for (int m = 0; m < X.ndims; ++m){
            int nextMode = arrX[m].modeOrder[1];

            sliceNnz[m] = arrX[m].nnzPerSlice[curIdx[m]];//+ arrX[nextMode].nnzPerSlice[curIdx[nextMode]];    ;
            // fbrNnz[m] =  arrX[nextMode].nnzPerSlice[curIdx[nextMode]];    
            //change to sameFm*m*
            if((m == 1 && sameFm0m1) || (m == 2 && sameFm1m2) || (m == 3 && sameFm2m3)){
                fbrNnz[m] = fbrNnz[m - 1];
                fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
            }
            else if((m == 2 && sameFm0m2) || (m == 3 && sameFm1m3)){
                fbrNnz[m] = fbrNnz[m - 2];
                fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
            }
            else if(m == 3 && sameFm0m3){
                fbrNnz[m] = fbrNnz[m - 3];
                fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
            }
            else{
                ITYPE result, tmp, tmpCounter = 0;
                ITYPE idx_j = curIdx[arrX[m].modeOrder[1]];
                tmpSlc = curIdx[m];

                /*linear search*/

                // for (int fbr = arrX[m].denseSlcPtr[tmpSlc]; fbr < arrX[m].denseSlcPtr[tmpSlc+1]; ++fbr){  

                //     if(arrX[m].fbrIdx[1][fbr] == idx_j){
                //         fbrNnz[m] = arrX[m].nnzPerFiber[fbr];
                //         fbrNo[m] = curIdx[arrX[m].modeOrder[1]];
                //         fbrSt[m] = fbr;
                //         break;
                //     }
                // }
                /*binary search*/
                {
                    int n =  arrX[m].denseSlcPtr[tmpSlc+1] - arrX[m].denseSlcPtr[tmpSlc];//sizeof(arr) / sizeof(arr[0]); 
                    ITYPE fbr = arrX[m].denseSlcPtr[tmpSlc];  
                    result = binarySearch1(&(arrX[m].fbrIdx[1][fbr]), 0, n, idx_j); 
                    tmp = arrX[m].nnzPerFiber[result+fbr];
                    fbrNnz[m] = tmp;
                }

                // if(idx%1000000 == 0 && idx < 10000000){
                //     cout << m<<" mm " << arrX[m].fbrPtr[0][tmpSlc]<<"," << arrX[m].denseSlcPtr[tmpSlc]
                //     <<" " <<arrX[m].fbrPtr[0][tmpSlc+1]<<"," << arrX[m].denseSlcPtr[tmpSlc+1]
                //     <<" from linear search " << fbrNnz[m]  <<" binary " << tmp << endl; //<<" hash"  << check
                // }
            }
            // sliceNnz[m] = fbrNnz[m];
            // sliceNnz[m] = arrX[m].nnzPerSlice[curIdx[m]] + 5*fbrNnz[m];
        }

        /* if fiber is longer */
        for (int m = 0; m < X.ndims-1; ++m){
            if(fbrNnz[m] != fbrNnz[m+1] )
                allsame = false;
        }

        // if (allsame) {
        //     allnnz1++;
        //     mode = shortestMode;
        //     modeDone = true;
        // }

        if(X.ndims == 3){

            if ( fbrNnz[0] >=  fbTh * std::max(fbrNnz[1] , fbrNnz[2]) && !modeDone) {
                modeDone = true;
                if(sameFm0m1 || sameFm0m2 || sameFm0m3)
                    mode = shortMode;
                else
                    mode = 0;
            }
            else if ( fbrNnz[1] >=  fbTh * std::max(fbrNnz[0] , fbrNnz[2]) && !modeDone) {
                modeDone = true;
                if(sameFm1m2 || sameFm1m3)
                    mode = shortMode;
                else 
                    mode = 1;
            }
            else if ( fbrNnz[2] >=  fbTh * std::max(fbrNnz[0] , fbrNnz[1]) && !modeDone) {
                modeDone = true;
                if(sameFm2m3)
                    mode = shortMode;
                else 
                    mode = 2;
            }

            else if( slcNnzPerParti[0][curIdx[0]] >= std::max(slcNnzPerParti[1][curIdx[1]] , slcNnzPerParti[2][curIdx[2]]) && !modeDone){
                mode = 0;
            }

            else if( slcNnzPerParti[1][curIdx[1]] >= std::max(slcNnzPerParti[0][curIdx[0]] , slcNnzPerParti[2][curIdx[2]]) && !modeDone){
                mode = 1;
            }

            else if( slcNnzPerParti[2][curIdx[2]] >= std::max(slcNnzPerParti[1][curIdx[1]] , slcNnzPerParti[0][curIdx[0]]) && !modeDone){
                mode = 2;
            }
        }

        else if(X.ndims == 4){

            if ( fbrNnz[0] >=  fbTh * maxOf3(fbrNnz[1] , fbrNnz[2], fbrNnz[3]) && !modeDone) {
                modeDone = true;
                if(sameFm0m1 || sameFm0m2 || sameFm0m3)
                    mode = shortMode;
                else
                    mode = 0;
            }
            else if ( fbrNnz[1] >=  fbTh * maxOf3(fbrNnz[0] , fbrNnz[2], fbrNnz[3]) && !modeDone) {
                modeDone = true;
                if(sameFm1m2 || sameFm1m3)
                    mode = shortMode;
                else 
                    mode = 1;
            }
            else if ( fbrNnz[2] >=  fbTh * maxOf3(fbrNnz[0] , fbrNnz[1], fbrNnz[3]) && !modeDone) {
                modeDone = true;
                if(sameFm2m3)
                    mode = shortMode;
                else 
                    mode = 2;
            }
            else if ( fbrNnz[3] >=  fbTh * maxOf3(fbrNnz[0] , fbrNnz[1], fbrNnz[2]) && !modeDone) {
                modeDone = true;
                mode = 3;
            }

            else if( slcNnzPerParti[0][curIdx[0]] >= maxOf3(slcNnzPerParti[1][curIdx[1]] , 
                slcNnzPerParti[2][curIdx[2]], slcNnzPerParti[3][curIdx[3]]) && !modeDone){
                mode = 0;
            }

            else if( slcNnzPerParti[1][curIdx[1]] >= maxOf3(slcNnzPerParti[0][curIdx[0]] , 
                slcNnzPerParti[2][curIdx[2]], slcNnzPerParti[3][curIdx[3]]) && !modeDone){
                mode = 1;
            }

            else if( slcNnzPerParti[2][curIdx[2]] >= maxOf3(slcNnzPerParti[1][curIdx[1]] , 
                slcNnzPerParti[0][curIdx[0]], slcNnzPerParti[3][curIdx[3]]) && !modeDone){
                mode = 2;
            }

            else if( slcNnzPerParti[3][curIdx[3]] >= maxOf3(slcNnzPerParti[1][curIdx[1]] , 
                slcNnzPerParti[0][curIdx[0]], slcNnzPerParti[2][curIdx[2]]) && !modeDone){
                mode = 3;
            }
        }

        slcNnzPerParti[mode][curIdx[mode]]++;

        if(!modeDone)
            mode = -1;

        // if(idx < X.totNnz/2)
        //     mode = 0;
        // else mode = 1;

        // if( mode == 2)
        //     for (int i = 0; i < 4; ++i)
        //     {
        //         cout << curIdx[i] << " ";
        //     }
        //     cout << endl;
        // // mode = 3;
    
        /*populate new partitions*/
        if(mode > -1){
            X.partPerNnz[idx] = mode;
        }
        if(casePr) 
            cout << "selected mode: " << mode << endl;
    }
    }
    cout << "nnz1 in all mode " << allnnz1 << endl;

    // for (int m = 0; m < X.ndims; ++m)
    //     MTX[m].totNnz = MTX[m].vals.size();
}
// more detailed check
// inline int mm_partition_reuseBased(Tensor *arrX, Tensor &X, TiledTensor *MTX, Options & Opt){

//     X.partPerNnz = new ITYPE[X.totNnz];
//     memset(X.partPerNnz, 0, X.totNnz * sizeof(ITYPE));  
//     X.totnnzPerPart = new ITYPE[X.ndims];
//     memset(X.totnnzPerPart, 0, X.ndims * sizeof(ITYPE));  
 
//     for (int m = 0; m < arrX[0].ndims; ++m){
        
//         int sliceMode=arrX[m].modeOrder[0];
//         int fiberMode=arrX[m].modeOrder[1];
        
//         arrX[m].nnzPerFiber = new ITYPE[arrX[m].nFibers];
//         memset(arrX[m].nnzPerFiber, 0, arrX[m].nFibers * sizeof(ITYPE));     

//         arrX[m].nnzPerSlice = new ITYPE[arrX[m].dims[sliceMode]];
//         memset(arrX[m].nnzPerSlice, 0, arrX[m].dims[sliceMode] * sizeof(ITYPE));  

//         arrX[m].denseSlcPtr = (ITYPE*)malloc( (arrX[m].dims[sliceMode]+1) * sizeof(ITYPE)); //new ITYPE[arrX[m].dims[sliceMode]];
//         memset(arrX[m].denseSlcPtr, 0, (arrX[m].dims[sliceMode] + 1 ) * sizeof(ITYPE));  
//     }

//     /*creating dense slices so that nnz can directly index slices unlike fiber. For
//     fiber it needs to scan all fibers in a slice. */

//     for (int m = 0; m < arrX[0].ndims; ++m){

//         // #pragma omp parallel 
//         {
//         // #pragma omp for 
//         for(ITYPE slc = 0; slc < arrX[m].fbrIdx[0].size(); ++slc) {

//             arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]] = arrX[m].fbrPtr[0][slc];
                   
//             if(slc == arrX[m].fbrIdx[0].size()-1)
//                 arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]+1] = arrX[m].fbrPtr[0][slc];
//             else
//                 arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]+1] = arrX[m].fbrPtr[0][slc+1];

//             /* Populate nnz per fiber and nnz per slice */
//             for (int fbr = arrX[m].fbrPtr[0][slc]; fbr < arrX[m].fbrPtr[0][slc+1]; ++fbr){      
               
//                 arrX[m].nnzPerFiber[fbr] = arrX[m].fbrPtr[1][fbr+1] - arrX[m].fbrPtr[1][fbr];
//                 arrX[m].nnzPerSlice[arrX[m].fbrIdx[0][slc]] += arrX[m].nnzPerFiber[fbr];
//             }
//         }
//         }
//     }

//     // #pragma omp barrier

//     int threshold = ( X.totNnz / X.dims[0] + X.totNnz / X.dims[1] + X.totNnz / X.dims[2]) / 3;
//     int thNnzInTile = X.totNnz*1;

//     /* initialize MICSF tiles */
//     int mode = 0;

//     for (int m = 0; m < X.ndims; ++m){

//         MTX[m].ndims = X.ndims;
//         MTX[m].dims = new ITYPE[MTX[m].ndims];  
//         MTX[m].totNnz = 0; // WHY mode?
        
//         for (int i = 0; i < X.ndims; ++i){
//             MTX[m].modeOrder.push_back(arrX[m].modeOrder[i]); 
//             MTX[m].dims[i] = X.dims[i];
//         }     
//     }    

//     /* Populate with nnz for each slice for each mode */

//     ITYPE mode0 = 0;//std::min(X.dims[0], X.dims[1], X.dims[2]);
//     ITYPE mode1 = 1;//X.modeOrder[1];
//     ITYPE mode2 = 2;//X.modeOrder[2];
//     ITYPE mode3 = 3;

//     // int *fbrNnz = new int[X.ndims];
//     // int *fbrNo = new int[X.ndims];
//     // int *fbrSt = new int[X.ndims];
//     // int *curIdx = new int[X.ndims];
//     // int *sliceNnz = new int[X.ndims];
//     // int tmpSlc;

//     bool sameFm0m1 = false, sameFm0m2 = false, sameFm1m2 = false, sameFm0m3 = false, sameFm1m3 = false, sameFm2m3 = false;

//     int fbTh =  Opt.MIfbTh;
//     int slTh =  1, shortMode = 0;
//     cout << "fiber threshold: " << fbTh << " MUST CHANGE last else " << endl;

//     for (int m = 0; m < X.ndims; ++m){

//         if(m == 1){
//             if (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
//                 sameFm0m1 = true;
//                 shortMode = (arrX[m].dims[0] <= arrX[m].dims[1] ? 0 : 1);
//             }
//         }
//         else if(m == 2){
//             if(arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]){
//                 sameFm0m2 = true;
//                 shortMode = (arrX[m].dims[0] <= arrX[m].dims[2] ? 0 : 2);
//             }
//             else if ( arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
//                 sameFm1m2 = true;
//                 shortMode = (arrX[m].dims[1] <= arrX[m].dims[2] ? 1 : 2);
//             }
//         }
//         else if(m == 3){
//             if(arrX[m].modeOrder[1] == arrX[m-3].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-3].modeOrder[1]){
//                 sameFm0m3 = true;
//                 shortMode = (arrX[m].dims[0] <= arrX[m].dims[3] ? 0 : 3);
//             }
//             else if ( arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]){
//                 sameFm1m3 = true;
//                 shortMode = (arrX[m].dims[1] <= arrX[m].dims[3] ? 1 : 3);
//             }
//             else if ( arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
//                 sameFm2m3 = true;
//                 shortMode = (arrX[m].dims[2] <= arrX[m].dims[3] ? 2 : 3);
//             }
//         }
//     }
//     bool casePr = false;
//     #pragma omp barrier

//     /******** Process NNZ********s*/
//     // #pragma omp parallel 
//     {
//     ITYPE *fbrNnz = new ITYPE[X.ndims];
//     memset(fbrNnz, 0, X.ndims * sizeof(ITYPE));     
//     ITYPE *fbrNo = new ITYPE[X.ndims];
//     ITYPE *curIdx = new ITYPE[X.ndims];
//     ITYPE *sliceNnz =  new ITYPE[X.ndims];
//     ITYPE tmpSlc;
//     int mode;

//     // #pragma omp for 
//     for (int idx = 0; idx < X.totNnz; ++idx){
        
//         bool modeDone = false;

//         // if(idx%1000000 == 0 && idx < 10000000 )  casePr = true;
//         // else casePr = false;
   
//         for (int m = 0; m < X.ndims; ++m)
//             curIdx[m] = X.inds[m][idx];

//         /*Finding fiber nnz*/
//         for (int m = 0; m < X.ndims; ++m){
//             int nextMode = arrX[m].modeOrder[1];

//             sliceNnz[m] = arrX[m].nnzPerSlice[curIdx[m]];//+ arrX[nextMode].nnzPerSlice[curIdx[nextMode]];    ;
//             // fbrNnz[m] =  arrX[nextMode].nnzPerSlice[curIdx[nextMode]];    
//             //change to sameFm*m*
//             if((m == 1 && sameFm0m1) || (m == 2 && sameFm1m2) || (m == 3 && sameFm2m3)){
//                 fbrNnz[m] = fbrNnz[m - 1];
//                 fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
//             }
//             else if((m == 2 && sameFm0m2) || (m == 3 && sameFm1m3)){
//                 fbrNnz[m] = fbrNnz[m - 2];
//                 fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
//             }
//             else if(m == 3 && sameFm0m3){
//                 fbrNnz[m] = fbrNnz[m - 3];
//                 fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
//             }
//             else{
//                 ITYPE result, tmp, tmpCounter = 0;
//                 ITYPE idx_j = curIdx[arrX[m].modeOrder[1]];
//                 tmpSlc = curIdx[m];

//                 /*linear search*/

//                 // for (int fbr = arrX[m].denseSlcPtr[tmpSlc]; fbr < arrX[m].denseSlcPtr[tmpSlc+1]; ++fbr){  

//                 //     if(arrX[m].fbrIdx[1][fbr] == idx_j){
//                 //         fbrNnz[m] = arrX[m].nnzPerFiber[fbr];
//                 //         fbrNo[m] = curIdx[arrX[m].modeOrder[1]];
//                 //         fbrSt[m] = fbr;
//                 //         break;
//                 //     }
//                 // }
//                 /*binary search*/
//                 {
//                     int n =  arrX[m].denseSlcPtr[tmpSlc+1] - arrX[m].denseSlcPtr[tmpSlc];//sizeof(arr) / sizeof(arr[0]); 
//                     ITYPE fbr = arrX[m].denseSlcPtr[tmpSlc];  
//                     result = binarySearch1(&(arrX[m].fbrIdx[1][fbr]), 0, n, idx_j); 
//                     tmp = arrX[m].nnzPerFiber[result+fbr];
//                     fbrNnz[m] = tmp;
//                 }

//                 // if(idx%1000000 == 0 && idx < 10000000){
//                 //     cout << m<<" mm " << arrX[m].fbrPtr[0][tmpSlc]<<"," << arrX[m].denseSlcPtr[tmpSlc]
//                 //     <<" " <<arrX[m].fbrPtr[0][tmpSlc+1]<<"," << arrX[m].denseSlcPtr[tmpSlc+1]
//                 //     <<" from linear search " << fbrNnz[m]  <<" binary " << tmp << endl; //<<" hash"  << check
//                 // }
//             }
//             // sliceNnz[m] = fbrNnz[m];
//             // sliceNnz[m] = arrX[m].nnzPerSlice[curIdx[m]] + 5*fbrNnz[m];
//         }
//         if(casePr) {
//             cout << curIdx[0] << " " << curIdx[1] <<" " << curIdx[2] << endl; 
//             for (int m = 0; m < X.ndims; ++m){

//                  cout << m <<" slcNnz: "<<sliceNnz[m] << " " <<" fbrNnz: (" <<curIdx[m]<<","<<
//                  fbrNo[m]  <<")- " << fbrNnz[m] << endl;
//             }
//         }

//         /* if fiber is longer */

//         // /*check fiber 0*/
//         if(sameFm0m1 || sameFm0m2 || !modeDone){

//             if(casePr) cout << "case 0 true" << endl;

//             if(sameFm0m1){
//                // fiberLen 0 and fiberLen 1 are same
//                 if ( fbrNnz[0] >=  fbTh * fbrNnz[2]) {
                    
//                     modeDone = true;
//                     mode = shortMode;
            
//                 }
//                 else if (fbrNnz[2] >=  fbTh * fbrNnz[0]) {
//                     modeDone = true;
//                     mode = 2;
//                 }
//             }
//             else if(sameFm0m2){

//                 if ( fbrNnz[0] >=  fbTh * fbrNnz[1]) {

//                      modeDone = true;
//                      mode = shortMode;
                
//                 }
//                 else if (fbrNnz[1] >=  fbTh * fbrNnz[0]) {
//                     modeDone = true;
//                     mode = 1;
//                 }
//             }
//         }

//         else if ( fbrNnz[0] >=  fbTh * std::max(fbrNnz[1] , fbrNnz[2]) && !modeDone) {
//             // add a alice condition maybe 2 * fbth
//             if(casePr ) cout << "case 1 true" << endl;
//             modeDone = true;
//             mode = 0;
//         }

//         /*check fiber 1*/
//         if(sameFm1m2 && !modeDone){ //m0m1 already taken care of in prev if

//             if(casePr ) cout << "case 2 true" << endl;
           
//             if ( fbrNnz[1] >=  fbTh * fbrNnz[0]) {
                
//                 modeDone = true;
//                 mode = shortMode;
//             }
//             else if (fbrNnz[0] >=  fbTh * fbrNnz[1]) {
//                 modeDone = true;
//                 mode = 0;
//             }           
//         }

//         else if ( fbrNnz[1] >=  fbTh * std::max(fbrNnz[0] , fbrNnz[2]) && !modeDone) {
//             modeDone = true;
//             mode = 1;
//             if(casePr ) cout << "case 3 true" << endl;
//         }

//         /*check fibe 2*/
//         //sameFm0m2, sameFm1m2 are taken care of
//         if ( fbrNnz[2] >=  fbTh * std::max(fbrNnz[0] , fbrNnz[1]) && !modeDone) {
//             mode = 2;
//             modeDone = true;
//             if(casePr ) cout << "case 5 true" << endl;           
//         }

//         /* if slice is longer */
//         if ( sliceNnz[2] >=  slTh * std::max(sliceNnz[0], sliceNnz[1]) && !modeDone) { 
//             mode = 2;//mode1;
//             modeDone = true;
//             if(casePr ) cout << "case 7 true" << endl;
//         }

//         else if ( sliceNnz[1] >=  slTh * std::max(sliceNnz[0], sliceNnz[2]) && !modeDone)    { 
//             mode = 1;//mode0;
//             modeDone = true;
//             if(casePr ) cout << "case 6 true" << endl;
//         }

//         else if ( sliceNnz[0] >=  slTh * std::max(sliceNnz[1], sliceNnz[2])  && !modeDone)  {   
//             modeDone = true;
//             mode = 0;//mode2;
//             if(casePr ) cout << "case 8 true" << endl;
//         }

//         if(!modeDone){
//             mode = -1;

//         }
//         // if(idx < 6)
//         //     mode = 1;
//         // else mode = 2;

//         // if( mode == 2)
//         //     for (int i = 0; i < 4; ++i)
//         //     {
//         //         cout << curIdx[i] << " ";
//         //     }
//         //     cout << endl;
//         // // mode = 3;
    
//         /*populate new partitions*/
//         if(mode > -1){
//             X.partPerNnz[idx] = mode;
//         }
//         if(casePr) 
//             cout << "selected mode: " << mode << endl;
//     }
//     }

//     // for (int m = 0; m < X.ndims; ++m)
//     //     MTX[m].totNnz = MTX[m].vals.size();
// }

// can take the slice nnz conditions
// inline int mm_partition_allMode(Tensor *arrX, Tensor &X, TiledTensor *MTX, Options & Opt){
 
//     X.partPerNnz = new ITYPE[X.totNnz];
//     memset(X.partPerNnz, 0, X.totNnz * sizeof(ITYPE));  
//     X.totnnzPerPart = new ITYPE[X.ndims];
//     memset(X.totnnzPerPart, 0, X.ndims * sizeof(ITYPE));  

//     for (int m = 0; m < arrX[0].ndims; ++m){
        
//         int sliceMode=arrX[m].modeOrder[0];
//         int fiberMode=arrX[m].modeOrder[1];
        
//         arrX[m].nnzPerFiber = new ITYPE[arrX[m].nFibers];
//         memset(arrX[m].nnzPerFiber, 0, arrX[m].nFibers * sizeof(ITYPE));     

//         arrX[m].nnzPerSlice = new ITYPE[arrX[m].dims[sliceMode]];
//         memset(arrX[m].nnzPerSlice, 0, arrX[m].dims[sliceMode] * sizeof(ITYPE));  

//         arrX[m].fiberPerSlice = new ITYPE[arrX[m].dims[sliceMode]];
//         memset(arrX[m].fiberPerSlice, 0, arrX[m].dims[sliceMode] * sizeof(ITYPE));  

//         arrX[m].denseSlcPtr = (ITYPE*)malloc( (arrX[m].dims[sliceMode]+1) * sizeof(ITYPE)); //new ITYPE[arrX[m].dims[sliceMode]];
//         memset(arrX[m].denseSlcPtr, 0, (arrX[m].dims[sliceMode] + 1 ) * sizeof(ITYPE));  
//     }

//     /*creating dense slices so that nnz can directly index slices unlike fiber. For
//     fiber it needs to scan all fibers in a slice. */

//     for (int m = 0; m < arrX[0].ndims; ++m){

//         for(ITYPE slc = 0; slc < arrX[m].fbrIdx[0].size(); ++slc) {

//             arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]] = arrX[m].fbrPtr[0][slc];

//             arrX[m].fiberPerSlice[arrX[m].fbrIdx[0][slc]] = arrX[m].fbrPtr[0][slc+1] - arrX[m].fbrPtr[0][slc];
                   
//             if(slc == arrX[m].fbrIdx[0].size()-1)
//                 arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]+1] = arrX[m].fbrPtr[0][slc];
//             else
//                 arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]+1] = arrX[m].fbrPtr[0][slc+1];
//         }
        
//     }

//     // Populate nnz per fiber and nnz per slice 
//     for (int m = 0; m < arrX[0].ndims; ++m) {

//         /*dont need vals or last ids*/
//         // arrX[m].vals.resize(0);
//         // arrX[m].inds[X.modeOrder[2]].resize(0);

//         for(ITYPE slc = 0; slc < arrX[m].fbrIdx[0].size(); ++slc) {

//             for (int fbr = arrX[m].fbrPtr[0][slc]; fbr < arrX[m].fbrPtr[0][slc+1]; ++fbr){      
               
//                 arrX[m].nnzPerFiber[fbr] = arrX[m].fbrPtr[1][fbr+1] - arrX[m].fbrPtr[1][fbr];
//                 arrX[m].nnzPerSlice[arrX[m].fbrIdx[0][slc]] += arrX[m].nnzPerFiber[fbr];
//             }
//         }
//     }

//     int threshold = ( X.totNnz / X.dims[0] + X.totNnz / X.dims[1] + X.totNnz / X.dims[2]) / 3;
//     int singleSliceFromAllMode;
//     int thNnzInTile = X.totNnz*1;


//     /* initialize MICSF tiles */
//     int mode = 0;

//     for (int m = 0; m < X.ndims; ++m){
//         bool switchBC = false;
//         ITYPE switchMode;
//         MTX[m].ndims = X.ndims;
//         MTX[m].dims = new ITYPE[MTX[m].ndims];  

//         //setting mode order accroding to mode length
//         int  mMode1 = (1 + m) % X.ndims;
//         int  mMode2 = (2 + m) % X.ndims;

//         if( X.dims[mMode1] > X.dims[mMode2]) switchBC = true; else false;  
        
//         for (int i = 0; i < X.ndims; ++i){
//             MTX[m].inds.push_back(std::vector<ITYPE>());  
//             MTX[m].dims[i] = X.dims[i];
//             MTX[mode].totNnz = 0;
//             // MTX[m].modeOrder.push_back((i+m) % X.ndims);

//             if(i > 0 && switchBC){

//                 if(i == 1) switchMode = 2;
//                 else if(i == 2) switchMode = 1;
//             }
//             else
//                 switchMode = i;       
//             MTX[m].modeOrder.push_back((m + switchMode) % X.ndims);
//         } 
//         // cout << m <<" " <<  MTX[m].modeOrder[0]  << " " <<  MTX[m].modeOrder[1]
//         // <<" " <<  MTX[m].modeOrder[2] << endl;        
//     }

//     /* Populate with nnz for each slice for each mode */

//     ITYPE mode0 = 0;//std::min(X.dims[0], X.dims[1], X.dims[2]);
//     ITYPE mode1 = 1;//X.modeOrder[1];
//     ITYPE mode2 = 2;//X.modeOrder[2];

//     int *fbrNnz = new int[X.ndims];
//     int *fbrNo = new int[X.ndims];
//     int *fbrSt = new int[X.ndims];
//     int *curIdx = new int[X.ndims];
//     int *sliceNnnz =  new int[X.ndims];
//     int tmpSlc;

//     bool sameFm0m1 = false, sameFm0m2 = false, sameFm1m2 = false;
//     int fbTh =  Opt.MIfbTh;
//     int slTh =  1;
//     cout << "threshold: " << fbTh <<","<< endl;

//     for (int m = 0; m < X.ndims; ++m){

//         if(m == 1){
//             if (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1])
//                 sameFm0m1 = true;
//         }
//         else if(m == 2){
//             if(arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1])
//                 sameFm0m2 = true;
//             else if ( arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1])
//                 sameFm1m2 = true;
//         }
//     }
//     bool casePr = false;

//     // /* Process NNZs*/
//     for (int idx = 0; idx < X.totNnz; ++idx){
        
//         bool modeDone = false;

//         // if(idx%1000000 == 0 && idx < 10000000 ) 
//         //     casePr = true;
//         // else
//         //      casePr = false;
   
//         for (int m = 0; m < X.ndims; ++m)
//             curIdx[m] = X.inds[m][idx];

//         /*Finding fiber nnz*/
//         for (int m = 0; m < X.ndims; ++m){

//             sliceNnnz[m] = arrX[m].fiberPerSlice[curIdx[m]];
//             //change to sameFm*m*
//             if((m == 1 && sameFm0m1) || (m == 2 && sameFm1m2)){
//                 fbrNnz[m] = fbrNnz[m - 1];
//                 fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
//             }
//             else if(m == 2 && sameFm0m2){
//                 fbrNnz[m] = fbrNnz[m - 2];
//                 fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
//             }
//             else{
//                 ITYPE result, tmp, tmpCounter = 0;
//                 ITYPE idx_j = curIdx[arrX[m].modeOrder[1]];
//                 tmpSlc = curIdx[m];

//                 /*linear search*/

//                 // for (int fbr = arrX[m].denseSlcPtr[tmpSlc]; fbr < arrX[m].denseSlcPtr[tmpSlc+1]; ++fbr){  

//                 //     if(arrX[m].fbrIdx[1][fbr] == idx_j){
//                 //         fbrNnz[m] = arrX[m].nnzPerFiber[fbr];
//                 //         fbrNo[m] = curIdx[arrX[m].modeOrder[1]];
//                 //         fbrSt[m] = fbr;
//                 //         break;
//                 //     }
//                 // }
//                 /*binary search*/
//                 {
//                     int n =  arrX[m].denseSlcPtr[tmpSlc+1] - arrX[m].denseSlcPtr[tmpSlc];//sizeof(arr) / sizeof(arr[0]); 
//                     ITYPE fbr = arrX[m].denseSlcPtr[tmpSlc];  
//                     result = binarySearch1(&(arrX[m].fbrIdx[1][fbr]), 0, n, idx_j); 
//                     tmp = arrX[m].nnzPerFiber[result+fbr];
//                     fbrNnz[m] = tmp;
//                 }

//                 // if(idx%1000000 == 0 && idx < 10000000){
//                 //     cout << m<<" mm " << arrX[m].fbrPtr[0][tmpSlc]<<"," << arrX[m].denseSlcPtr[tmpSlc]
//                 //     <<" " <<arrX[m].fbrPtr[0][tmpSlc+1]<<"," << arrX[m].denseSlcPtr[tmpSlc+1]
//                 //     <<" from linear search " << fbrNnz[m]  <<" binary " << tmp << endl; //<<" hash"  << check
//                 // }
//             }
//         }
//         if(casePr) {
//             cout << curIdx[0] << " " << curIdx[1] <<" " << curIdx[2] << endl; 
//             for (int m = 0; m < X.ndims; ++m){

//                  cout << m <<" slcNnz: "<<sliceNnnz[m] << " " <<" fbrNnz: (" <<curIdx[m]<<","<<
//                  fbrNo[m]  <<")- " << fbrNnz[m] << endl;
//             }
//         }

//         /* if fiber is longer */
//         /*check fiber 0*/
//         if(sameFm0m1 || sameFm0m2 || !modeDone){

//             if(casePr) cout << "case 0 true" << endl;

//             if(sameFm0m1){
//                // fiberLen 0 and fiberLen 1 are same
//                 if ( fbrNnz[0] >=  fbTh * fbrNnz[2]) {
                    
//                     modeDone = true;
                    
//                     if(sliceNnnz[0] >=  slTh * sliceNnnz[1] ) 
//                         mode = 0;             
//                     else 
//                         mode = 1;//arrX[0].modeOrder[1] ;  
//                 }
//                 else if (fbrNnz[2] >=  fbTh * fbrNnz[0]) {
//                     modeDone = true;
//                     mode = 2;
//                 }
//             }
//             else if(sameFm0m2){

//                 if ( fbrNnz[0] >=  fbTh * fbrNnz[1]) {

//                      modeDone = true;
                    
//                     if(sliceNnnz[0] >=  slTh * sliceNnnz[2] ) 
//                         mode = 0;
//                     else 
//                         mode = 2;//arrX[0].modeOrder[1] ;
//                 }
//                 else if (fbrNnz[1] >=  fbTh * fbrNnz[0]) {
//                     modeDone = true;
//                     mode = 1;
//                 }
//             }
//         }

//         else if ( fbrNnz[0] >=  fbTh * std::max(fbrNnz[1] , fbrNnz[2]) && !modeDone) {
//             // add a alice condition maybe 2 * fbth
//             if(casePr ) cout << "case 1 true" << endl;
//             modeDone = true;
//             mode = 0;
//         }

//         /*check fiber 1*/
//         if(sameFm1m2 && !modeDone){ //m0m1 already taken care of in prev if

//             if(casePr ) cout << "case 2 true" << endl;
           
//             if ( fbrNnz[1] >=  fbTh * fbrNnz[0]) {
//                 modeDone = true;
//                 if(sliceNnnz[1] >=  slTh * sliceNnnz[2] ) mode = 1;
//                 else mode = 2;//arrX[1].modeOrder[1] ;
//             }
//             else if (fbrNnz[0] >=  fbTh * fbrNnz[1]) {
//                 modeDone = true;
//                 mode = 0;
//             }           
//         }

//         else if ( fbrNnz[1] >=  fbTh * std::max(fbrNnz[0] , fbrNnz[2]) && !modeDone) {
//             modeDone = true;
//             mode = 1;
//             if(casePr ) cout << "case 3 true" << endl;
//         }

//         /*check fibe 2*/
//         //sameFm0m2, sameFm1m2 are taken care of
//         if ( fbrNnz[2] >=  fbTh * std::max(fbrNnz[0] , fbrNnz[1]) && !modeDone) {
//             mode = 2;
//             modeDone = true;
//             if(casePr ) cout << "case 5 true" << endl;           
//         }

//         /* if slice is longer */
//         if ( sliceNnnz[1] >=  slTh * std::max(sliceNnnz[0], sliceNnnz[2]) && !modeDone)    { 
//             mode = 1;//mode0;
//             modeDone = true;
//             if(casePr ) cout << "case 6 true" << endl;
//         }

//         else if ( sliceNnnz[2] >=  slTh * std::max(sliceNnnz[0], sliceNnnz[1]) && !modeDone) { 
//             mode = 2;//mode1;
//             modeDone = true;
//             if(casePr ) cout << "case 7 true" << endl;
//         }
   
//         else if ( sliceNnnz[0] >=  slTh * std::max(sliceNnnz[1], sliceNnnz[2])  && !modeDone)  {   
//             modeDone = true;
//             mode = 0;//mode2;
//             if(casePr ) cout << "case 8 true" << endl;
//         }

//         if(!modeDone)
//             mode = -1;
    
//         /*populate new partitions*/
//         if(mode > -1){
//             X.partPerNnz[idx] = mode;
//             // for (int i = 0; i < X.ndims; ++i)  {
//             //     MTX[mode].inds[i].push_back(X.inds[i][idx]); 
//             // }
//             // MTX[mode].vals.push_back(X.vals[idx]); 
//             // MTX[mode].totNnz = MTX[mode].vals.size();
//         }
//         if(casePr) 
//             cout << "selected mode: " << mode << endl;

//     }

//     // for (int m = 0; m < X.ndims; ++m)
//     //     MTX[m].totNnz = MTX[m].vals.size();
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
  
    for (int m = 0; m < X.ndims; ++m){  
        mode = X.modeOrder[m];
        srand48(0L);
        for(long r = 0; r < U[mode].nRows; ++r){
            for(long c = 0; c < U[mode].nCols; ++c) // or u[mode].nCols 
                U[mode].vals[r * U[mode].nCols + c] = mode + .5;//1.5 * (mode+1);;// .1 * drand48(); //1 ;//; //
        }
    }
    return 0;
}

inline int zero_mat(const Tensor &X, Matrix *U, ITYPE mode){
    
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
    for (int i = 0; i < 3; ++i)
    // for (int i = U[mode].nRows-5; i <  U[mode].nRows; ++i)
    {
        // for (int j = 0; j < U[mode].nCols; ++j)
        for (int j = 0; j < 3; ++j)
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
                if(mismatch < 5 && j == 0)
                  cout << "mismatch at (" << i <<"," << j <<") got: " << out[i * nc +j] << " exp: " << COOout[i * nc +j] << ". "<< endl;
                mismatch++;
                // exit(0);
            }          
        }
    }

    if(mismatch == 0)
        cout << "Correctness pass!" << endl;
    else{
        cout <<  mismatch <<" mismatches found at " << precision << " precision. " << endl;
        cout << "Maximum diff " << maxDiff << ". "<<endl;
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

        case 'b':
            param.TBsize = atoi(argv[i]);
            break;

        case 's':
            param.fiberPerWarp = atoi(argv[i]);
            break;

        case 'h':
            param.MIfbTh = atoi(argv[i]);
            break;

        case 'g':
            param.gridSize = atoi(argv[i]);
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
        
        case 'p':
            param.m0 = argv[i];
            break;
        
        case 'q':
            param.m1 = argv[i];
            break;

        case 'r':
            param.m2 = argv[i];
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


