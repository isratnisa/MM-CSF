#include <mpi.h>
#include "util.h"

#define mpi_barrier() MPI_Barrier(MPI_COMM_WORLD);

class MPI_param{
    public:
    	int mpi_rank, n_proc;

};

int start_mpi(MPI_param &MPIparam){

	MPI_Init(NULL, NULL);
    
    // Find out rank, size
 
    int root_p = sqrt(MPIparam.n_proc+1);
    int row_color = MPIparam.mpi_rank / root_p;
    int col_color = MPIparam.mpi_rank % root_p;
    MPI_Comm row_comm, col_comm;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &MPIparam.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPIparam.n_proc);
    MPI_Comm_split(MPI_COMM_WORLD, row_color, MPIparam.mpi_rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col_color, MPIparam.mpi_rank, &col_comm);

    int row_rank, row_size, col_rank, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    // printf("WORLD RANK/SIZE: %d/%d \t ROW RANK/SIZE: %d/%d %d/%d\n", MPIparam.mpi_rank, MPIparam.n_proc, row_rank, row_size,  col_rank, col_size);
}

int create_mpi_partition(const Tensor &X, const MPI_param &MPIparam){

	int idealNnzSize = X.totNnz/MPIparam.n_proc;

	cout << idealNnzSize << " boo " << endl;

	int n_proc = MPIparam.n_proc;

	int *mpiNoSlc = new int [n_proc];
	int *mpiNoFbr = new int [n_proc];
	int *mpiNoNnz = new int [n_proc];

	int *mpiStSlc = new int [n_proc];
	int *mpiEndSlc = new int [n_proc];
	int *mpiStFbr = new int [n_proc];
	int *mpiEndFbr = new int [n_proc];
	int *mpiStNnz = new int [n_proc];
	int *mpiEndNnz = new int [n_proc];

	int *nnzPerSlc = new int [X.fbrIdx[0].size()];

	long nnzSoFar = 0;

	/*Partition should be based on slice. You want to keep all fibers of the same slice together. 
	Otherwise, multiple gpu will update same slice but at their own node */
	int partition = 0;

	for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

		nnzPerSlc[slc] = 0;

        for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){   

			nnzPerSlc[slc] += X.fbrPtr[1][fbr+1] - X.fbrPtr[1][fbr];
		}
	}
	mpi_barrier();

	mpiStSlc[0] = 0;

	for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

		nnzSoFar +=  nnzPerSlc[slc];
		// if(MPIparam.mpi_rank == 0)
		// cout << slc << " nnzsofar : " <<  nnzSoFar <<" " << nnzPerSlc[slc]  << endl;

		if(nnzSoFar > idealNnzSize || slc == X.fbrIdx[0].size() - 1){

			if(slc == X.fbrIdx[0].size() - 1){

				if(MPIparam.mpi_rank == 0)
				cout << "I am in else " << endl;
				
				mpiEndSlc[partition] = X.fbrIdx[0].size();
				mpiEndFbr[partition] = X.fbrIdx[1].size();
				mpiEndNnz[partition] = X.totNnz;
			}

			else{
				if(MPIparam.mpi_rank == 0)
				cout << "I am in if " << endl;

				mpiEndSlc[partition] = slc;
				int endfiber = X.fbrPtr[0][slc+1];
				mpiEndFbr[partition] = endfiber;
				mpiEndNnz[partition] = X.fbrPtr[1][endfiber];
				nnzSoFar = 0;		
				partition++;
			}
		}
    }

    mpi_barrier();
	
	if(MPIparam.mpi_rank == 0)
    
    for (int i = 0; i < n_proc; ++i)
    {
    	cout << i << " rank : " <<  mpiEndSlc[i] << " " << mpiEndFbr[i] 
    	 << " " << mpiEndNnz[i] << endl; 
    }
    MPI_Finalize();
}

