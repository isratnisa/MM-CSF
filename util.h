#ifndef UTIL_H
#define UTIL_H

#define DTYPE float
#define ITYPE unsigned int

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
        ITYPE nnz;
        ITYPE nFibers;
        ITYPE *accessK;
        bool switchBC = false;
        std::vector<ITYPE> modeOrder;
		std::vector<vector<ITYPE>> inds;
		std::vector<float> vals;
        std::vector<ITYPE> slicePtr;
        std::vector<ITYPE> sliceIdx;
        std::vector<ITYPE> fiberPtr;
        std::vector<ITYPE> fiberIdx;
        std::vector<vector<ITYPE>> slcMapperBin;
        std::vector<vector<ITYPE>> rwBin;
};

class HYBTensor{
    public:
        ITYPE ndims;
        ITYPE *dims;
        ITYPE nnz;
        ITYPE COOnnz;
        ITYPE CSLnnz;
        ITYPE nFibers;
        ITYPE *accessK;
        bool switchBC = false;
        std::vector<ITYPE> modeOrder;
        std::vector<vector<ITYPE>> inds;
        std::vector<float> vals;
        std::vector<ITYPE> slicePtr;
        std::vector<ITYPE> sliceIdx;
        std::vector<ITYPE> fiberPtr;
        std::vector<ITYPE> fiberIdx;
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
        ITYPE nnz;
        ITYPE nFibers;
        ITYPE *accessK;
        bool switchBC = false;     // if true change matrix rand() to 1
        std::vector<ITYPE> modeOrder;
        std::vector<vector<ITYPE>> inds;
        std::vector<float> vals;
        std::vector<ITYPE> slicePtr;
        std::vector<ITYPE> sliceIdx;
        std::vector<ITYPE> fiberPtr;
        std::vector<ITYPE> fiberIdx;
        std::vector<vector<ITYPE>> slcMapperBin;
        std::vector<vector<ITYPE>> rwBin;
};

class Matrix{
    public:
        ITYPE nRows;
        ITYPE nCols;
        //vector<float> vals;
        float *vals;
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
    float vid=0;
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
		fp >> X.dims[switchMode]; 
        X.modeOrder.push_back((switchMode + Opt.mode) % X.ndims);
		X.inds.push_back(std::vector<ITYPE>());
	}

	while(fp >> index) {
        X.inds[0].push_back(index-1);
		for (int i = 1; i < X.ndims; ++i)
		{      
			fp >> index;
            if(X.switchBC && i > 0 ){
                if(i == 1)
                    switchMode = 2;
                else if(i == 2)
                    switchMode = 1;
            }
            else
                switchMode = i;
			X.inds[switchMode].push_back(index-1);   
		}
		fp >> vid;
		X.vals.push_back(vid);

	}
    X.nnz = X.vals.size();

	// for (int i = 0; i < X.nnz; ++i)
	// {
	// 	cout << X.inds[0][i] << " " << X.inds[1][i] << " "<< X.inds[2][i] <<endl;
	// }
    //    cout << "nnz " << X.nnz << endl;
    return 0;
}

inline int compute_accessK(Tensor &X, const Options &Opt){

    ITYPE mode2 = X.modeOrder[2];
    X.accessK = new ITYPE[X.dims[mode2]];
    memset(X.accessK, 0, X.dims[mode2] * sizeof(ITYPE));
    
    for(ITYPE x = 0; x < X.nnz; ++x) {
    
       ITYPE idx2 = X.inds[mode2][x];
       X.accessK[idx2]++;
    } 
    // for (int i = 0; i <  X.dims[mode2]; ++i)
    // {
    //     cout << i <<": " << X.accessK[i] << endl; 
    // }
}

inline int create_write_heavy(Tensor &X, const Options &Opt){

    int shLimit = 192;
    int nnzSlc = 0;
    int nRwBin = 3;

    for (int b = 0; b < nRwBin; ++b)
    {
        X.rwBin.push_back(std::vector<ITYPE>());
    }
    cout <<  X.rwBin[0].size() <<" " <<  X.rwBin[1].size() << endl;
    for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {

        for (int fbr = X.slicePtr[slc]; fbr < X.slicePtr[slc+1]; ++fbr){              
            nnzSlc += X.fiberPtr[fbr+1] - X.fiberPtr[fbr]; 
        }

        //for now just write only bin
        //bin 0 write heavy
        //bin 1 ready heavy
        //bin 3 equal == COO
        if (nnzSlc > shLimit) {
            X.rwBin[0].push_back(slc);
        }
        else
            X.rwBin[1].push_back(slc);      
    }
    return 0;
}

inline int print_COOtensor(const Tensor &X){

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];

    for(ITYPE x = 0; x < X.nnz; ++x) {
    
        cout << X.inds[mode0][x] << " " << X.inds[mode1][x] << " " << X.inds[mode2][x] << endl;

    }           
}

inline int print_HCSRtensor(const Tensor &X){

    cout << "no of fibers " << X.fiberPtr.size() << endl;

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];

    for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {

        ITYPE idx0 = X.sliceIdx[slc];
        int fb_st = X.slicePtr[slc];
        int fb_end = X.slicePtr[slc+1];
        printf("slc st- end: %d %d %d \n", slc, fb_st, fb_end );
        
        for (int fbr = fb_st; fbr < fb_end; ++fbr){        
             printf("fbr %d :  ", fbr );    
            ITYPE idx1 = X.fiberIdx[fbr];
            for(ITYPE x = X.fiberPtr[fbr]; x < X.fiberPtr[fbr+1]; ++x) {
                if(mode0 == 0)
                    cout << idx0 << " " << idx1 << " " << X.inds[2][x] << endl;
                if(mode0 == 1)
                    cout  << idx1 << " " << X.inds[1][x] << " "<< idx0 << endl;
                if(mode0 == 2)
                    cout  << X.inds[0][x]  << " " << idx0 << " " << idx1<< endl;

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
    
    cout << "HCSR " <<HybX.sliceIdx.size() << endl;

    for(ITYPE slc = 0; slc < HybX.sliceIdx.size(); ++slc) {

        ITYPE idx0 = HybX.sliceIdx[slc];
        int fb_st = HybX.slicePtr[slc];
        int fb_end = HybX.slicePtr[slc+1];
        printf("slc st- end: %d %d %d \n", slc, fb_st, fb_end );
        
        for (int fbr = fb_st; fbr < fb_end; ++fbr){        
             printf("fbr %d :  ", fbr );    
            ITYPE idx1 = HybX.fiberIdx[fbr];
            for(ITYPE x = HybX.fiberPtr[fbr]; x < HybX.fiberPtr[fbr+1]; ++x) {
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

    cout << "no of fibers " << TiledX[tile].fiberPtr.size() << endl;
    
    for(ITYPE slc = 0; slc < TiledX[tile].sliceIdx.size(); ++slc) {

        ITYPE idx0 = TiledX[tile].sliceIdx[slc]; //slc
        int fb_st = TiledX[tile].slicePtr[slc];
        int fb_end = TiledX[tile].slicePtr[slc+1];
        // printf("slc st- end: %d %d %d \n", slc, fb_st, fb_end );
        
        for (int fbr = fb_st; fbr < fb_end; ++fbr){        
            // printf("fbr %d :  ", fbr );    
            for(ITYPE x = TiledX[tile].fiberPtr[fbr]; x < TiledX[tile].fiberPtr[fbr+1]; ++x) {
                cout << idx0 << " " << TiledX[tile].inds[1][x] << " " << TiledX[tile].inds[2][x] << endl;

            }            
        }
    }
}

inline int make_KTiling(const Tensor &X, TiledTensor *TiledX, const Options &Opt){

    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];
    
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

    for (int idx = 0; idx < X.nnz; ++idx){

        tile = X.inds[mode2][idx]/Opt.tileSize;

        for (int i = 0; i < X.ndims; ++i)  {
            TiledX[tile].inds[i].push_back(X.inds[i][idx]); 
        }

        TiledX[tile].vals.push_back(X.vals[idx]);      
    }

    for (int tile = 0; tile < Opt.nTile; ++tile){
        TiledX[tile].nnz = TiledX[tile].vals.size();
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

//  creating pointers and fiber indices for 3D tensors
inline int create_HCSR(Tensor &X, const Options &Opt){

    ITYPE fbrThreashold = Opt.fbrThreashold;
    
    bool useHYB = true;
    ITYPE sliceId, fiberId;
    ITYPE mode0 = X.modeOrder[0];
    ITYPE mode1 = X.modeOrder[1];
    ITYPE mode2 = X.modeOrder[2];

    std::vector<ITYPE> tmpFbrPtr;
    std::vector<ITYPE> tmpFbrIdx;

    ITYPE prevSliceId =  X.inds[mode0][0];
    ITYPE prevFiberId =  X.inds[mode1][0];

    X.slicePtr.push_back(0);
    X.fiberPtr.push_back(0);
    X.sliceIdx.push_back(prevSliceId);
    X.fiberIdx.push_back(prevFiberId);
    
    int idx = 1 ;
    
    while(idx < X.nnz) {
        
        sliceId = X.inds[mode0][idx];
        fiberId = X.inds[mode1][idx];   
   
        ITYPE fiberNnz = 1;
        while( fiberId == prevFiberId && sliceId == prevSliceId && idx < X.nnz && fiberNnz < fbrThreashold){
            ++idx;
            fiberNnz++;
            sliceId = X.inds[mode0][idx];
            fiberId = X.inds[mode1][idx];      
        }
        if(idx == X.nnz)
            break;
        
        X.fiberPtr.push_back(idx);
        X.fiberIdx.push_back(fiberId);
        
        if( sliceId != prevSliceId) {//not else ..not become this in loop
            X.sliceIdx.push_back(sliceId);
            X.slicePtr.push_back((ITYPE)(X.fiberPtr.size()) - 1);
        }      
        prevSliceId = sliceId;
        prevFiberId = fiberId;
        ++idx;
        fiberNnz = 1;
    }
    X.fiberPtr.push_back(idx);
    X.fiberIdx.push_back(fiberId);
    X.slicePtr.push_back((ITYPE)(X.fiberPtr.size() -1 ));
    X.nFibers = X.fiberPtr.size() - 1;

    return 0;
}

inline int create_HYB(HYBTensor &HybX, const Tensor &X, const Options &Opt){

    ITYPE fbrThreashold = Opt.fbrThreashold;
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

    for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {
        sliceNnz = 0;
        fbrLenOne = true;

        for (int fbr = X.slicePtr[slc]; fbr < X.slicePtr[slc+1]; ++fbr){  
            fiberNnz = X.fiberPtr[fbr+1] - X.fiberPtr[fbr]; 
            if(fiberNnz > 1) fbrLenOne = false;    
            sliceNnz += fiberNnz;
        }
        // cout << slc << " " << sliceNnz << endl;
        int stFiber =  X.slicePtr[slc];
        int endFiber =  X.slicePtr[slc+1];
        
        if(sliceNnz == 1){       
            int idx =  X.fiberPtr[X.slicePtr[slc]];
            HybX.COOinds[mode0].push_back(slc);
            HybX.COOinds[mode1].push_back(X.fiberIdx[stFiber]);
            HybX.COOinds[mode2].push_back(X.inds[mode2][idx]); 
            HybX.COOvals.push_back(X.vals[idx]);  
            usedCOOSlc++;
            usedCOOFbr++;
        
        }
        else if(fbrLenOne) {    
            HybX.CSLslicePtr.push_back(X.slicePtr[slc] - (usedCOOFbr + usedHCSRFbr));
            HybX.CSLsliceIdx.push_back(X.sliceIdx[slc]);    
            for (int fbr = X.slicePtr[slc]; fbr < X.slicePtr[slc+1]; ++fbr){ 
                int idx =  X.fiberPtr[fbr];    
                HybX.CSLinds[mode1].push_back(X.fiberIdx[fbr]);
                HybX.CSLinds[mode2].push_back(X.inds[mode2][idx]); 
                HybX.CSLvals.push_back(X.vals[idx]);  
                
            }
            usedCSLFbr +=  X.slicePtr[slc + 1] - X.slicePtr[slc];
            usedCSLSlc++;
        }
        else{

            HybX.slicePtr.push_back(X.slicePtr[slc] - (usedCOOFbr + usedCSLFbr));
            HybX.sliceIdx.push_back(X.sliceIdx[slc]);
            
            for (int fbr = X.slicePtr[slc]; fbr < X.slicePtr[slc+1]; ++fbr){   
                
                HybX.fiberPtr.push_back(X.fiberPtr[fbr] - (usedCOOFbr + usedCSLFbr));   
                HybX.fiberIdx.push_back(X.fiberIdx[fbr]); 
                copy(X.inds[mode2].begin() + X.fiberPtr[fbr] , X.inds[mode2].begin() + X.fiberPtr[fbr+1], std::back_inserter(HybX.inds[mode2]));       
                copy(X.vals.begin() + X.fiberPtr[fbr] , X.vals.begin() + X.fiberPtr[fbr+1], std::back_inserter(HybX.vals));          
            }
            usedHCSRFbr += X.slicePtr[slc + 1] - X.slicePtr[slc];
            usedHCSRSlc++;
        }
    }
    
    HybX.fiberPtr.push_back(HybX.inds[mode2].size());
    HybX.slicePtr.push_back((ITYPE)(HybX.fiberPtr.size() -1 ));
    HybX.CSLslicePtr.push_back((ITYPE)(HybX.CSLvals.size()));
    HybX.nFibers = HybX.fiberPtr.size() - 1;
    HybX.COOnnz = HybX.COOvals.size();
    HybX.CSLnnz = HybX.CSLvals.size();
    HybX.nnz = HybX.vals.size();
    if(Opt.verbose){
        cout << "slices in COO " <<HybX.COOnnz << endl;
        cout << "slices in CSL " <<HybX.CSLsliceIdx.size() << endl;
        cout << "slices in HCSR " <<HybX.sliceIdx.size() << endl;
    }
    return 0;
}
// inline int create_HYB(HYBTensor &HybX, const Tensor &X, const Options &Opt){

//     ITYPE fbrThreashold = Opt.fbrThreashold;
//     bool fbrLenOne = true;

//     ITYPE sliceId, fiberId, sliceNnz = 0, fiberNnz = 0;
//     int nCOOSlc = 0, nFbr = 0, nCOOFbr = 0, nCLSFbr = 0, nHCSRFbr = 0;
    
//     ITYPE mode0 = HybX.modeOrder[0];
//     ITYPE mode1 = HybX.modeOrder[1];
//     ITYPE mode2 = HybX.modeOrder[2];

//     for (int i = 0; i < X.ndims; ++i){
//          HybX.COOinds.push_back(std::vector<ITYPE>()); 
//          HybX.inds.push_back(std::vector<ITYPE>());
//      }

//     for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {
//         sliceNnz = 0;
//         fbrLenOne = true;

//         for (int fbr = X.slicePtr[slc]; fbr < X.slicePtr[slc+1]; ++fbr){  
//             fiberNnz = X.fiberPtr[fbr+1] - X.fiberPtr[fbr]; 
//             if(fiberNnz > 1) fbrLenOne = false;    
//             sliceNnz += fiberNnz;
//         }
//         // cout << slc << " " << sliceNnz << endl;
//         int stFiber =  X.slicePtr[slc];
//         int endFiber =  X.slicePtr[slc+1];
        
//         if(sliceNnz == 1){       
//             int idx =  X.fiberPtr[X.slicePtr[slc]];
//             HybX.COOinds[0].push_back(slc);
//             HybX.COOinds[1].push_back(X.fiberIdx[stFiber]);
//             HybX.COOinds[2].push_back(X.inds[mode2][idx]); 
//             HybX.COOvals.push_back(X.vals[idx]);  
//             nCOOSlc++;
//             nFbr += endFiber - stFiber;
        
//         }
//         // else if(fbrLenOne)  {     
//         //     HybX.CSLslicePtr.push_back(X.slicePtr[slc]-nCOOSlc);
//         //     HybX.CSLsliceIdx.push_back(X.sliceIdx[slc]);    
//         //     for (int fbr = X.slicePtr[slc]; fbr < X.slicePtr[slc+1]; ++fbr){    
//         //         HybX.fiberPtr.push_back(X.fiberPtr[fbr] - nFbr);   
//         //         HybX.fiberIdx.push_back(X.fiberIdx[fbr]); 
//         //         copy(X.inds[mode2].begin() + X.fiberPtr[fbr] , X.inds[mode2].begin() + X.fiberPtr[fbr+1], std::back_inserter(HybX.inds[mode2]));       
//         //         copy(X.vals.begin() + X.fiberPtr[fbr] , X.vals.begin() + X.fiberPtr[fbr+1], std::back_inserter(HybX.vals));       
//         //         nFbr += endFiber - stFiber;
//         //     }
//         // }
//         else{

//             HybX.slicePtr.push_back(X.slicePtr[slc]-nCOOSlc);
//             HybX.sliceIdx.push_back(X.sliceIdx[slc]);
            
//             for (int fbr = X.slicePtr[slc]; fbr < X.slicePtr[slc+1]; ++fbr){    
//                 HybX.fiberPtr.push_back(X.fiberPtr[fbr] - nFbr);   
//                 HybX.fiberIdx.push_back(X.fiberIdx[fbr]); 
//                 copy(X.inds[mode2].begin() + X.fiberPtr[fbr] , X.inds[mode2].begin() + X.fiberPtr[fbr+1], std::back_inserter(HybX.inds[mode2]));       
//                 copy(X.vals.begin() + X.fiberPtr[fbr] , X.vals.begin() + X.fiberPtr[fbr+1], std::back_inserter(HybX.vals));       
//                 // nFbr += endFiber - stFiber;
//             }
//         }
//     }
    
//     HybX.fiberPtr.push_back(HybX.inds[mode2].size());
//     HybX.slicePtr.push_back((ITYPE)(HybX.fiberPtr.size() -1 ));
//     HybX.nFibers = HybX.fiberPtr.size() - 1;
//     HybX.COOnnz = HybX.COOvals.size();
    
//     cout << "slices in COO " <<HybX.COOnnz << endl;
//     cout << "slices in HCSR " <<HybX.sliceIdx.size() << endl;
//     return 0;
// }
//  creating pointers and fiber indices for 3D tensors
// Replace all X by TiledX[tile].. otherwise no diff

inline int create_TiledHCSR(TiledTensor *TiledX, const Options &Opt , int tile){

    ITYPE sliceId, fiberId;
    ITYPE mode0 = TiledX[tile].modeOrder[0];
    ITYPE mode1 = TiledX[tile].modeOrder[1];
    ITYPE mode2 = TiledX[tile].modeOrder[2];
    ITYPE fbrThreashold = Opt.fbrThreashold;
    
    TiledX[tile].slicePtr.push_back(0);
    TiledX[tile].fiberPtr.push_back(0);
    ITYPE prevSliceId =  TiledX[tile].inds[mode0][0];
    ITYPE prevFiberId =  TiledX[tile].inds[mode1][0];
    TiledX[tile].sliceIdx.push_back(prevSliceId);
    TiledX[tile].fiberIdx.push_back(prevFiberId);
    
    int idx = 1 ;
    
    while(idx < TiledX[tile].nnz) {
        
        sliceId = TiledX[tile].inds[mode0][idx];
        fiberId = TiledX[tile].inds[mode1][idx];   
   
        ITYPE fiberNnz = 1;
        while( fiberId == prevFiberId && sliceId == prevSliceId && idx < TiledX[tile].nnz && fiberNnz < fbrThreashold){
            ++idx;
            fiberNnz++;
            sliceId = TiledX[tile].inds[mode0][idx];
            fiberId = TiledX[tile].inds[mode1][idx];           
        }
        if(idx == TiledX[tile].nnz)
            break;
        TiledX[tile].fiberPtr.push_back(idx);
        TiledX[tile].fiberIdx.push_back(fiberId);
        
        if( sliceId != prevSliceId) {//not else ..not become this in loop
            TiledX[tile].sliceIdx.push_back(sliceId);
            TiledX[tile].slicePtr.push_back((ITYPE)(TiledX[tile].fiberPtr.size()) - 1);
        }      
        prevSliceId = sliceId;
        prevFiberId = fiberId;
        ++idx;
        fiberNnz = 1;
    }
    TiledX[tile].fiberPtr.push_back(idx);
    TiledX[tile].fiberIdx.push_back(fiberId);
    TiledX[tile].slicePtr.push_back((ITYPE)(TiledX[tile].fiberPtr.size() -1 ));
    TiledX[tile].nFibers = TiledX[tile].fiberPtr.size() - 1;

    return 0;
}
// changed param to HYB
inline int make_Bin(HYBTensor &X, const Options & Opt){

    ITYPE THREADLOAD = 2;
    ITYPE TB = 512;
    std::vector<ITYPE> UB;
    std::vector<ITYPE> LB;

    // Bin boundaries
    for (int i = 0; i < Opt.nBin; i++) {
        X.slcMapperBin.push_back(std::vector<ITYPE>());
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
    LB[9] = 32 * TB ;   UB[9] = X.nnz + 1;  // 512 WARP = 32 TB

    UB[Opt.nBin - 1] = X.nnz + 1;
    
    // Populate HCSR bin
    for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {
        // int slc = X.sliceIdx[slcPtr];
        int fb_st = X.slicePtr[slc];
        int fb_end = X.slicePtr[slc+1];
        int nnzSlc = 0;

        for (int fbr = fb_st; fbr < fb_end; ++fbr){              
            nnzSlc += X.fiberPtr[fbr+1] - X.fiberPtr[fbr]; 
        }
        // #pragma omp parallel
        // {
        // unsigned int cpu_thread_id = omp_get_thread_num();
        // int i = cpu_thread_id;
        for (int bin = 0; bin < Opt.nBin; ++bin)
        {
            // cout << bin << " " << LB[bin] <<" " << UB[bin] << endl;
            if (nnzSlc > LB[bin] && nnzSlc < UB[bin]) {
                X.slcMapperBin[bin].push_back(slc);
            }
        }
    }

    // // Populate CSL bin
    // for(ITYPE slc = 0; slc < X.CSLsliceIdx.size(); ++slc) {
    //     // int slc = X.sliceIdx[slcPtr];
    //     int fb_st = X.CSLslicePtr[slc];
    //     int fb_end = X.CSLslicePtr[slc+1];
    //     int nnzSlc = 0;

    //     for (int fbr = fb_st; fbr < fb_end; ++fbr){              
    //         nnzSlc += fb_end - fb_st; 
    //     }
    //     // #pragma omp parallel
    //     // {
    //     // unsigned int cpu_thread_id = omp_get_thread_num();
    //     // int i = cpu_thread_id;
    //     for (int bin = 0; bin < Opt.nBin; ++bin)
    //     {
    //         // cout << bin << " " << LB[bin] <<" " << UB[bin] << endl;
    //         if (nnzSlc > LB[bin] && nnzSlc < UB[bin]) {
    //             X.CSLslcMapperBin[bin].push_back(slc);
    //         }
    //     }
    // }
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
    LB[9] = 32 * TB ;   UB[9] = TiledX[tile].nnz + 1;  // 512 WARP = 32 TB

    UB[Opt.nBin - 1] = TiledX[tile].nnz + 1;
    
    // Populate bin
    for(ITYPE slc = 0; slc < TiledX[tile].sliceIdx.size(); ++slc) {
        // int slc = TiledX[tile].sliceIdx[slcPtr];
        int fb_st = TiledX[tile].slicePtr[slc];
        int fb_end = TiledX[tile].slicePtr[slc+1];
        int nnzSlc = 0;

        for (int fbr = fb_st; fbr < fb_end; ++fbr){              
            nnzSlc += TiledX[tile].fiberPtr[fbr+1] - TiledX[tile].fiberPtr[fbr]; 
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
            }
        }
    }

    // debug

    // for (int i = 0; i < Opt.nBin; ++i)    {
    //     if(TiledX[tile].slcMapperBin[i].size() > 0){
    //         cout << "bin "<< i << ": "<< TiledX[tile].slcMapperBin[i].size() << endl;

    //         // for (int j = 0; j < TiledX[tile].slcMapperBin[i].size(); ++j)
    //         // {
    //         //     cout << TiledX[tile].sliceIdx[TiledX[tile].slcMapperBin[i][j]] << " ";
    //         // }
    //         cout << endl;
    //     }
    // }
}

inline int tensor_stats(const Tensor &X){

    ITYPE mode0 = X.modeOrder[0];
    int *nnzSlice = new int[X.sliceIdx.size()];
    int *nnzFibers = new int[X.nFibers];
    ITYPE totNnz = 0, flopsSaved = 0, emptySlc = 0;
    ITYPE minSlcNnz = 999999999, maxSlcNnz = 0;
    double stdDev = 0, stdDevFbr = 0;
    int avgSlcNnz = X.nnz/X.dims[mode0]; // int to use stdDev
    int avgFbrNnz = X.nnz/X.nFibers;

    // ofstream ofslc("slc_info.txt");
 
    for(ITYPE slc = 0; slc < X.sliceIdx.size(); ++slc) {
        nnzSlice[slc] = 0;

        ITYPE idx0 = slc;
        int fb_st = X.slicePtr[slc];
        int fb_end = X.slicePtr[slc+1];
        // int nnzInFiber = fb_end - fb_st;
        
        for (int fbr = fb_st; fbr < fb_end; ++fbr){   
            int nnzInFiber = X.fiberPtr[fbr+1] - X.fiberPtr[fbr]; 
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
        //ofslc << slc << " nFiber: " << X.slicePtr[slc+1]- X.slicePtr[slc] <<" nnzSlice: "<< nnzSlice[slc] 
        //<< " avg: "<< nnzSlice[slc] /(X.slicePtr[slc+1]- X.slicePtr[slc]) << endl;

    }
    cout << "flopsSaved " << " ,emptySlc " <<" ,minSlcNnz " << " ,maxSlcNnz " << " ,avgSlcNnz " << " ,stdDvSlcNnz "<< " ,stdDvFbrNnz " << ",nFibers ";
    cout << endl;
    
    cout << flopsSaved;
    cout << ", " << emptySlc <<", " << minSlcNnz << ", " << maxSlcNnz;
    cout << ", " << avgSlcNnz << ", "<< sqrt(stdDev/X.dims[mode0]) << ", "<< sqrt(stdDevFbr/X.nFibers);
    cout << ", " << X.nFibers << ", " ;//<< X.rwBin[0].size() << ", " << X.rwBin[1].size();
    cout << endl;

    if(totNnz == X.nnz)
        cout << "nnz matched " << totNnz << endl;
    else
        cout << "nnz not matched! sliceNnz " << totNnz << " X.nnz " << X.nnz << endl;

    return 0;
}

inline int create_mats(const Tensor &X, Matrix *U, const Options &Opt){
    
    ITYPE mode;
    ITYPE R = Opt.R;
    for (int m = 0; m < X.ndims; ++m){  
        mode = X.modeOrder[m];
        U[mode].nRows =  X.dims[mode];
        U[mode].nCols =  R;
        U[mode].vals = (float*)malloc(X.dims[mode] * R * sizeof(float));
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
                U[mode].vals[r * U[mode].nCols + c] = 0.1*drand48(); //1 ;//(r * R + c + 1); //
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


inline void correctness_check(Matrix *U, ITYPE mode,  string outFile, string seqFile){
    
    ifstream fo(outFile); 
    ifstream fs(seqFile); 
    int dummy, dummy1;
    string dumo, dums;
    long mismatch = 0;
    float precision = 0.05;
    
    std::getline(fo, dumo);
    std::getline(fs, dums);
    cout << "it has error !" << endl; 
    if(dumo.compare(dums) != 0){
        cout <<"dimension not matched! " << dumo <<" " << dums<< endl;
        exit(0);
    }

    cout << "matched" << endl;

    for (int i = 0; i < U[mode].nRows; ++i)
    {
       // #pragma omp parallel for
        for (int j = 0; j < U[mode].nCols; ++j)
        {
            float a,b;
            fo >>  a;
            fs >>  b;

            if(a - b > precision){
                cout << "mismatch at " << i *  U[mode].nCols + j <<" got: "
                << a << " exp: " << b << endl;
                mismatch++;
                // exit(0);
            }          
        }
    }
    if(mismatch == 0)
        cout << "Correctness pass!" << endl;
    else
        cout <<  mismatch <<" mismatches found at " << precision << " precision" << endl;
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


