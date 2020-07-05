#include <mpi.h>
#include "util.h"

#define mpi_barrier() MPI_Barrier(MPI_COMM_WORLD);

int start_mpi(MPI_param &MPIparam){

	MPI_Init(NULL, NULL);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &MPIparam.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPIparam.n_proc);

    printf("WORLD RANK/SIZE: %d/%d \n", MPIparam.mpi_rank, MPIparam.n_proc);
}
int end_mpi(){

	MPI_Finalize();
}

int create_mpi_partition(TiledTensor *MTX, int m,  const MPI_param &MPIparam){

	int idealNnzSize = MTX[m].totNnz/MPIparam.n_proc;

	cout << idealNnzSize << " boo " << endl;
	 mpi_barrier();

	int n_proc = MPIparam.n_proc;

	int *mpiNoSlc = new int [n_proc];
	int *mpiNoFbr = new int [n_proc];
	int *mpiNoNnz = new int [n_proc];

	int *mpiStSlc = new int [n_proc];
	int *mpiStFbr = new int [n_proc];
	int *mpiStNnz = new int [n_proc];
	
	MTX[m].mpiEndSlc = new ITYPE [n_proc];
	MTX[m].mpiEndFbr = new ITYPE [n_proc];
	MTX[m].mpiEndNnz = new ITYPE [n_proc];

	MTX[m].nnzInRank = new ITYPE [n_proc];
	MTX[m].fbrInRank = new ITYPE [n_proc];
	MTX[m].slcInRank = new ITYPE [n_proc];

	int *nnzPerSlc = new int [MTX[m].fbrIdx[0].size()];

	long nnzSoFar = 0;

	/*Partition should be based on slice. You want to keep all fibers of the same slice together. 
	Otherwise, multiple gpu will update same slice but at their own node */
	/*TBD:: For MM-CSF, partition can be based on fibers. */
	int partition = 0;

	for(ITYPE slc = 0; slc < MTX[m].fbrIdx[0].size(); ++slc) {

		nnzPerSlc[slc] = 0;

        for (int fbr = MTX[m].fbrPtr[0][slc]; fbr < MTX[m].fbrPtr[0][slc+1]; ++fbr){   

			nnzPerSlc[slc] += MTX[m].fbrPtr[1][fbr+1] - MTX[m].fbrPtr[1][fbr];
		}
	}
	mpi_barrier();

	mpiStSlc[0] = 0;

	for(ITYPE slc = 0; slc < MTX[m].fbrIdx[0].size(); ++slc) {

		nnzSoFar +=  nnzPerSlc[slc];
		// if(MPIparam.mpi_rank == 0)
		// 	cout << slc << " nnzsofar : " <<  nnzSoFar <<" " << nnzPerSlc[slc]  << endl;

		if(nnzSoFar > idealNnzSize || slc == MTX[m].fbrIdx[0].size() - 1){

			if(slc == MTX[m].fbrIdx[0].size() - 1){

				while(partition != n_proc){
					
					MTX[m].mpiEndSlc[partition] = MTX[m].fbrIdx[0].size();
					MTX[m].mpiEndFbr[partition] = MTX[m].fbrIdx[1].size();
					MTX[m].mpiEndNnz[partition] = MTX[m].totNnz;
					partition++;
				}
			}

			else{

				MTX[m].mpiEndSlc[partition] = slc;
				int endfiber = MTX[m].fbrPtr[0][slc+1];
				MTX[m].mpiEndFbr[partition] = endfiber;
				MTX[m].mpiEndNnz[partition] = MTX[m].fbrPtr[1][endfiber];
				nnzSoFar = 0;		
				partition++;
			}
		}
    }

    MTX[m].nnzInRank[0] = MTX[m].mpiEndNnz[0];
    MTX[m].fbrInRank[0] = MTX[m].mpiEndFbr[0];
    MTX[m].slcInRank[0] = MTX[m].mpiEndSlc[0];

    for (int p = 1; p < n_proc; ++p){

    	MTX[m].nnzInRank[p] = MTX[m].mpiEndNnz[p] - MTX[m].mpiEndNnz[p-1];
    	MTX[m].fbrInRank[p] = MTX[m].mpiEndFbr[p] - MTX[m].mpiEndFbr[p-1];    	
    	MTX[m].slcInRank[p] = MTX[m].mpiEndSlc[p] - MTX[m].mpiEndSlc[p-1];
    }

    mpi_barrier();
    
    if(MPIparam.mpi_rank == 1)
    for (int i = 0; i < n_proc; ++i)
    {
    	cout << "Partition: " << m << " rank: " << i<<" - " <<  MTX[m].mpiEndSlc[i] << " " << MTX[m].mpiEndFbr[i] 
    	 << " " << MTX[m].mpiEndNnz[i] << endl; 
    }
     mpi_barrier();
}

