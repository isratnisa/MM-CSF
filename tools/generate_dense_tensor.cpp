// g++ -O3 -std=c++11 -g -fopenmp generate_dense_tensor.cpp  &&./a.out test 10 10 10

#include <iostream>
#include <fstream>
#include <iomanip> 
using namespace std;

int main(int argc, char* argv[])
{
    cout << "Only generates 3D dense tensors" << endl;
   string outFile = argv[1];
   int *dims = new int [argc -1 ];
   int ndim =  argc -2;
   long nnz =1;

   for (int i = 0; i < ndim; ++i){
       
       dims[i] = atoi(argv[i+2]);
       nnz *=  dims[i];
   }
    cout << "Estimated nnz: " << nnz << endl;

    srand48(0L);

    ofstream fp(outFile); 

    fp << std::fixed;

    fp << ndim << endl;
    for (int i = 0; i < ndim; ++i)
        fp << dims[i] << "\t";
    fp << endl;

    
    for (int m0 = 0; m0 < dims[0]; ++m0){
        
        for (int m1 = 0; m1 < dims[1]; ++m1){

            for (int m2 = 0; m2 < dims[2]; ++m2)
        
                fp << std::setprecision(2) << m0+1 << "\t" << m1+1 << "\t" << m2+1 << "\t" << .1 * drand48() << endl;
        }     
    }
    return 0;
}