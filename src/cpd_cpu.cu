/**
 *
 * OHIO STATE UNIVERSITY SOFTWARE DISTRIBUTION LICENSE
 *
 * Load-balanced sparse MTTKRP on GPUs (the “Software”) Copyright (c) 2019, The Ohio State
 * University. All rights reserved.
 *
 * The Software is available for download and use subject to the terms and
 * conditions of this License. Access or use of the Software constitutes acceptance
 * and agreement to the terms and conditions of this License. Redistribution and
 * use of the Software in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the capitalized paragraph below.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the capitalized paragraph below in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. The names of Ohio State University, or its faculty, staff or students may not
 * be used to endorse or promote products derived from the Software without
 * specific prior written permission.
 *
 * THIS SOFTWARE HAS BEEN APPROVED FOR PUBLIC RELEASE, UNLIMITED DISTRIBUTION. THE
 * SOFTWARE IS PROVIDED “AS IS” AND WITHOUT ANY EXPRESS, IMPLIED OR STATUTORY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, WARRANTIES OF ACCURACY, COMPLETENESS,
 * NONINFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  ACCESS OR USE OF THE SOFTWARE IS ENTIRELY AT THE USER’S RISK.  IN
 * NO EVENT SHALL OHIO STATE UNIVERSITY OR ITS FACULTY, STAFF OR STUDENTS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  THE SOFTWARE
 * USER SHALL INDEMNIFY, DEFEND AND HOLD HARMLESS OHIO STATE UNIVERSITY AND ITS
 * FACULTY, STAFF AND STUDENTS FROM ANY AND ALL CLAIMS, ACTIONS, DAMAGES, LOSSES,
 * LIABILITIES, COSTS AND EXPENSES, INCLUDING ATTORNEYS’ FEES AND COURT COSTS,
 * DIRECTLY OR INDIRECTLY ARISING OUT OF OR IN CONNECTION WITH ACCESS OR USE OF THE
 * SOFTWARE.
 *
 */

/**
 *
 * Author:
 *          Israt Nisa (nisa.1@osu.edu)
 *
 * Contacts:
 *          Israt Nisa (nisa.1@osu.edu)
 *          Jiajia Li (jiajia.li@pnnl.gov)
 *
 */

#include "cpd_cpu.h"
// #include "cblas.h"
// #include "lapacke.h"
// #include "clapack.h"
#include "clapack.h"
#include <omp.h>


/* ATA only stores lower triangle elements. */
int MatrixDotMulSeqLowerTriangle(ITYPE const mode, ITYPE const ndims, Matrix * ATA, Matrix & tmpATA)
{
    assert(ATA[0].nRows == ATA[0].nCols);
    ITYPE const rank = ATA[0].nRows;

    DTYPE * ovals = tmpATA.vals;
    for(ITYPE i=0; i < rank; ++i) {
        for(ITYPE j=0; j < rank; ++j) {
            ovals[j * rank + i] = 1.0;
        }
    }

    for(ITYPE m=1; m < ndims; ++m) {
        ITYPE const pm = (mode + m) % ndims;
        DTYPE const * vals = ATA[pm].vals;
        for(ITYPE i=0; i < rank; ++i) {
            for(ITYPE j=0; j <= i; ++j) {
                ovals[i * rank + j] *= vals[i * rank + j];
            }
        }
    }

    /* Copy lower triangle to upper part */
    for(ITYPE i=0; i < rank; ++i) {
        for(ITYPE j=i+1; j < rank; ++j) {
            ovals[i * rank + j] = ovals[j * rank + i];
        }
    }
    
    return 0;
}


/* ATA only stores upper triangle elements. */
int MatrixDotMulSeqUpperTriangle(ITYPE const mode, ITYPE const ndims, Matrix * ATA, Matrix & tmpATA)
{
    assert(ATA[0].nRows == ATA[0].nCols);
    ITYPE const rank = ATA[0].nRows;

    DTYPE * ovals = tmpATA.vals;
    #pragma omp parallel
    {
      #pragma omp for schedule(static, 1)
      for(ITYPE i=0; i < rank; ++i) {
          for(ITYPE j=0; j < rank; ++j) {
              ovals[j * rank + i] = 1.0;
          }
      }

      for(ITYPE m=1; m < ndims; ++m) {
          ITYPE const pm = (mode + m) % ndims;
          DTYPE const * vals = ATA[pm].vals;
          #pragma omp for schedule(static, 1)
          for(ITYPE i=0; i < rank; ++i) {
              for(ITYPE j=i; j <= rank; ++j) {
                  ovals[i * rank + j] *= vals[i * rank + j];
              }
          }
      }

      #pragma omp barrier

      /* Copy lower triangle to upper part */
      #pragma omp for schedule(static, 1)
      for(ITYPE i=0; i < rank; ++i) {
          for(ITYPE j=0; j < i; ++j) {
              ovals[i * rank + j] = ovals[j * rank + i];
          }
      }
    } /* parallel region */
    
    return 0;
}


int MatrixSolveNormals(
  ITYPE const mode,
  ITYPE const ndims,
  Matrix * ATA,
  Matrix & tmpATA,
  Matrix & rhs)
{
  int rank = (int)(ATA[0].nCols);
  // ofstream fu, fa;
  // fu.open("U.txt", ofstream::app);
  // fa.open("ata.txt", ofstream::app);

  double t0 = seconds();
  MatrixDotMulSeqUpperTriangle(mode, ndims, ATA, tmpATA);

    // printf("  \t   MatrixDotMulSeqTriangle = %.3f \n", seconds() - t0);
  // fa << "tmpATA after MatrixDotMulSeqTriangle: " << endl;
  // write_output(&tmpATA, 0, "ata.txt");

  int info;
  char uplo = 'L';
  char tran_uplo = 'U';
  int nrhs = (int) rhs.nRows;
  DTYPE * const neqs = tmpATA.vals;

  t0 = seconds();
  /* Cholesky factorization */
  bool is_spd = true;
  spotrf_(&uplo, &rank, neqs, &rank, &info);
  if(info) {
    printf("Gram matrix is not SPD. Trying `gesv`.\n");
    is_spd = false;
  }
  // fa << "tmpATA after spotrf_: " << endl;
  // write_output(&tmpATA, 0, "ata.txt");

  /* Continue with Cholesky */
  if(is_spd) {
    /* Solve against rhs */
    spotrs_(&uplo, &rank, &nrhs, neqs, &rank, rhs.vals, &rank, &info);
    if(info) {
      printf("DPOTRS returned %d\n", info);
    }
    // fu << "U[m] after spotrs_: " << endl;
    // write_output(&rhs, 0, "U.txt");
  } 
  else {
    int * ipiv = (int*)malloc(rank * sizeof(int));  

    /* restore gram matrix */
    MatrixDotMulSeqUpperTriangle(mode, ndims, ATA, tmpATA);
    // fa << "tmpATA after MatrixDotMulSeqTriangle recover: " << endl;
    // write_output(&tmpATA, 0, "ata.txt");

    sgesv_(&rank, &nrhs, neqs, &rank, ipiv, rhs.vals, &rank, &info);
    // info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, rank, nrhs, neqs, rank, ipiv, rhs.vals, rank);

    if(info) {
      printf("sgesv_ returned %d\n", info);
    }
    // fu << "U[m] after sgesv_: " << endl;
    // write_output(&rhs, 0, "U.txt");

    free(ipiv);
  }
    // printf("  \t   solver = %.3f \n", seconds() - t0);

  return 0;
}


// Row-major
int Matrix2Norm(Matrix &A, DTYPE * lambda)
{
  ITYPE const nrows = A.nRows;
  ITYPE const ncols = A.nCols;
  DTYPE * const vals = A.vals;
  int nthreads = 1;
  #pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }

  DTYPE ** mylambda = (DTYPE **)malloc(nthreads * sizeof(DTYPE*));
  for(int i = 0; i < nthreads; ++i)
    mylambda[i] = (DTYPE *) malloc (ncols * sizeof(DTYPE));

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    for(ITYPE j=0; j < ncols; ++j) {
        mylambda[tid][j] = 0.0;
    }

    #pragma omp for schedule(static)
    for(ITYPE j=0; j < ncols; ++j) {
      lambda[j] = 0.0;
    }

    #pragma omp for schedule(static)
    for(ITYPE i=0; i < nrows; ++i) {
        for(ITYPE j=0; j < ncols; ++j) {
            mylambda[tid][j] += vals[i*ncols + j] * vals[i*ncols + j];
        }
    }
  }

  #pragma omp parallel for schedule(static)
    for(ITYPE j=0; j < ncols; ++j) {
      for(int i = 0; i < nthreads; ++i) {
      lambda[j] += mylambda[i][j];
    }
  }

  #pragma omp parallel for schedule(static)
  for(ITYPE j=0; j < ncols; ++j) {
      lambda[j] = sqrt(lambda[j]);
  }

  #pragma omp for schedule(static)
  for(ITYPE i=0; i < nrows; ++i) {
      for(ITYPE j=0; j < ncols; ++j) {
          vals[i*ncols + j] /= lambda[j];
      }
  }

    return 0;
}


// Row-major
int MatrixMaxNorm(Matrix &A, DTYPE * const lambda)
{
  ITYPE const nrows = A.nRows;
  ITYPE const ncols = A.nCols;
  DTYPE * const vals = A.vals;
  int nthreads = 1;
  #pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }

  DTYPE ** mylambda = (DTYPE **)malloc(nthreads * sizeof(DTYPE*));
  for(int i = 0; i < nthreads; ++i)
    mylambda[i] = (DTYPE *) malloc (ncols * sizeof(DTYPE));

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    for(ITYPE j=0; j < ncols; ++j) {
        mylambda[tid][j] = 0.0;
    }

    #pragma omp for schedule(static)
    for(ITYPE j=0; j < ncols; ++j) {
      lambda[j] = 0.0;
    }

    #pragma omp for schedule(static)
    for(ITYPE i=0; i < nrows; ++i) {
        for(ITYPE j=0; j < ncols; ++j) {
            if(vals[i*ncols + j] > lambda[j])
                mylambda[tid][j] = vals[i*ncols + j];
        }
    }
  }

  #pragma omp parallel for schedule(static)
    for(ITYPE j=0; j < ncols; ++j) {
      for(int i = 0; i < nthreads; ++i) {
      lambda[j] += mylambda[i][j];
    }
  }


  #pragma omp parallel for schedule(static)
  for(ITYPE j=0; j < ncols; ++j) {
      if(lambda[j] < 1)
          lambda[j] = 1;
  }

  #pragma omp parallel for schedule(static)
  for(ITYPE i=0; i < nrows; ++i) {
      for(ITYPE j=0; j < ncols; ++j) {
          vals[i*ncols + j] /= lambda[j];
      }
  }

  free(mylambda);

  return 0;
}

void GetFinalLambda(
  ITYPE const rank,
  ITYPE const ndims,
  Matrix * U,
  DTYPE * const lambda)
{
  DTYPE * tmp_lambda =  (DTYPE *) malloc(rank * sizeof(*tmp_lambda));

  for(ITYPE m=0; m < ndims; ++m) {   
    Matrix2Norm(U[m], tmp_lambda);
    for(ITYPE r=0; r < rank; ++r) {
      lambda[r] *= tmp_lambda[r];
    }
  }

  free(tmp_lambda);
}


double SparseTensorFrobeniusNormSquared(Tensor & X) 
{
  double norm = 0;
  // std::vector<DTYPE> vals = X.vals;
  
  for(ITYPE n=0; n < X.totNnz; ++n) {
    norm += X.vals[n] * X.vals[n];
  }
  return norm;
}

// Row-major. 
/* Compute a Kruskal tensor's norm is compute on "ATA"s. Check Tammy's sparse  */
double KruskalTensorFrobeniusNormSquared(
  ITYPE const ndims,
  DTYPE * lambda,
  Matrix * ATA,
  Matrix & tmpATA) // ATA: column-major
{
  ITYPE const rank = ATA[0].nCols;
  DTYPE * const __restrict tmp_atavals = tmpATA.vals; 
  double norm_mats = 0;

  for(ITYPE x=0; x < rank*rank; ++x) {
    tmp_atavals[x] = 1.;
  }

  /* Compute Hadamard product for all "ATA"s */
  for(ITYPE m=0; m < ndims; ++m) {
    DTYPE const * const __restrict atavals = ATA[m].vals;
    for(ITYPE i=0; i < rank; ++i) {
        for(ITYPE j=0; j <= i; ++j) {
            tmp_atavals[i * rank + j] *= atavals[i * rank + j];
        }
    }
  }

  /* compute lambda^T * ATA[MAX_NMODES] * lambda, only compute a half of them because of its symmetric */
  for(ITYPE i=0; i < rank; ++i) {
    norm_mats += tmp_atavals[i+(i*rank)] * lambda[i] * lambda[i];
    for(ITYPE j=0; j < i; ++j) {
      norm_mats += tmp_atavals[j+(i*rank)] * lambda[i] * lambda[j] * 2;
    }
  }

  return fabs(norm_mats);
}


// Row-major, compute via MTTKRP result (mats[nmodes]) and mats[nmodes-1].
double SparseKruskalTensorInnerProduct(
  ITYPE const ndims,
  DTYPE const * const __restrict lambda,
  Matrix * U,
  Matrix & tmpU) 
{
  const ITYPE rank = U[0].nCols;
  const ITYPE last_mode = ndims - 1;
  const ITYPE nrows = U[last_mode].nRows;
  
  DTYPE const * const last_vals = U[last_mode].vals;
  DTYPE const * const tmp_vals = tmpU.vals;
  double inner = 0;

  #pragma omp parallel reduction(+:inner)
  {
    int const tid = omp_get_thread_num();
    double * const __restrict accum = (double *) malloc(rank*sizeof(*accum));

    for(ITYPE r=0; r < rank; ++r) {
      accum[r] = 0.0; 
    }

    #pragma omp for
    for(ITYPE i=0; i < nrows; ++i) {
      for(ITYPE r=0; r < rank; ++r) {
        accum[r] += last_vals[r+(i*rank)] * tmp_vals[r+(i*rank)];
      }
    }

    for(ITYPE r=0; r < rank; ++r) {
      inner += accum[r] * lambda[r];
    }
  }

  return inner;
}



double KruskalTensorFit(
  Tensor & X,
  DTYPE * __restrict lambda,
  Matrix * U,
  Matrix & tmpU,
  Matrix * ATA,
  Matrix & tmpATA,
  double X_normsq) 
{
  ITYPE const ndims = X.ndims;

  double const norm_mats = KruskalTensorFrobeniusNormSquared(ndims, lambda, ATA, tmpATA);
  // printf("norm_mats: %lf\n", norm_mats);
  double const inner = SparseKruskalTensorInnerProduct(ndims, lambda, U, tmpU);
  // printf("inner: %lf\n", inner);
  double residual = X_normsq + norm_mats - 2 * inner;
  if (residual > 0.0) {
    residual = sqrt(residual);
  }
  // printf("residual: %lf\n", residual);
  double fit = 1 - (residual / sqrt(X_normsq));

  return fit;
}

#if 0
int dummy_func(){
  float a[] = {1, 2,  3,  4}; //NO need for column-major mode
  float b[] = {19, 22, 43, 50, 60 ,70}; //NO need for column-major mode
  int n = 2;
  int nrhs = 3;
  int lda = n;
  int ipiv[n];
  int ldb = n;
 
  int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, nrhs);
  
  cout << "From dummy_func(): error code " << info << endl;
  return 0;
}
#endif

double cpd(Tensor &X, TiledTensor *MMCSF, Matrix * U, Options Opt, DTYPE * lambda)
{
   // dummy_func();
    double fit = 0;
    ITYPE ndims = X.ndims;
    ITYPE * dims = X.dims;
    ITYPE rank = U[0].nCols;
    Matrix tmpU;
    ITYPE max_dim = 0;
    int niters = Opt.cpdIters;
    double tol = 1e-5;

    for(ITYPE m = 0; m < ndims; ++m) {
        if (dims[m] > max_dim)
            max_dim = dims[m];
    }     
    tmpU.nRows = max_dim;
    tmpU.nCols = rank;
    tmpU.vals = (DTYPE*) malloc(max_dim * rank * sizeof(DTYPE));

    //*** CPD:: CTC * BTB * ...
    Matrix * ATA = new Matrix[ndims];
    create_mats(X, ATA, Opt, true);
    Matrix tmpATA;
    tmpATA.nRows = rank;
    tmpATA.nCols = rank;
    tmpATA.vals = (DTYPE*) malloc(rank * rank * sizeof(DTYPE));    

    DTYPE alpha = 1.0, beta = 0.0;
    // CBLAS_TRANSPOSE cnotrans = CblasNoTrans;
    // CBLAS_TRANSPOSE ctrans = CblasTrans;
    // CBLAS_UPLO cuplo = CblasLower;
    char notrans = 'N';
    char trans = 'T';
    char uplo = 'L';
    int blas_rank = (int) rank;

    // Debug
    double t0 = 0;
    double t_cpd = 0;
    // string ata_file = "ata.txt";
    // string U_file = "U.txt";
    // ofstream fu, fa;
    // fu.open(U_file, ofstream::app);
    // fa.open(ata_file, ofstream::app);

    // fu << "Init U: " << endl;
    // for(ITYPE m = 0; m < ndims; ++m) {
    //   write_output(U, m, U_file);
    // }

    t0 = seconds();
    for(ITYPE m = 0; m < ndims; ++m) {
        int blas_nrows = (int)(U[m].nRows);
        // cblas_ssyrk(CblasRowMajor, cuplo, ctrans, 
        //     blas_rank, // The output size
        //     blas_nrows, // The reduce dim size
        //     alpha, U[m].vals, blas_rank, 
        //     beta, ATA[m].vals, blas_rank);
        ssyrk_(&uplo, &notrans, 
            &blas_rank, &blas_nrows, 
            &alpha, U[m].vals, &blas_rank, 
            &beta, ATA[m].vals, &blas_rank);
    }
    printf("  \t Init ssyrk = %.3f \n", seconds() - t0);

    double oldfit = 0;
    double total_mttkrp_time = 0;
    double X_normsq = SparseTensorFrobeniusNormSquared(X);
    // printf("X_normsq: %lf\n", X_normsq);
    /*GPU pinters - for reuse declaring here*/
    ITYPE *dInds2, *dfbrPtr1, *dfbrIdx1, *dFbrLikeSlcInds;
    DTYPE *dVals, *dU;
    t0 = seconds();
    init_GPU(MMCSF, U, Opt, &dInds2, &dfbrPtr1, &dfbrIdx1, &dFbrLikeSlcInds, &dVals, &dU);
    printf("  \t Init GPU = %.3f \n", seconds() - t0);
    unsigned int *szDU =  new unsigned int[X.ndims];
  
    for (int m = 0; m < X.ndims; ++m)
      szDU[m] = U[m].nRows * U[m].nCols;

    ITYPE loc = 0;

    t_cpd = seconds();
    for(int it=0; it < niters; ++it) {
        double cpd_t0 = seconds();

        for(ITYPE m=0; m < ndims; ++m) {
           if(Opt.verbose)
              cout << "\nit " << it << ", mode " << m << ":" << endl;
            // fa << "\nit " << it << ", mode " << m << ":" << endl;
            // fu << "\nit " << it << ", mode " << m << ":" << endl;
            // fa << "Init ATA: " << endl;
            // for(ITYPE m = 0; m < ndims; ++m) {
            //   write_output(ATA, m, ata_file);
            // }
            loc = 0;
            for (int mm = 0; mm < m; ++mm)
              loc += szDU[mm];

            t0 = seconds();
            Opt.mode = m;
            X.modeOrder[0] = m;
            for(ITYPE i=1; i<ndims; ++i)
              X.modeOrder[i] = (m+i) % ndims; 
            tmpU.nRows = U[m].nRows;
            /* Initialize U[mode] */
            // zero_mat(X, U, m);
           if(Opt.verbose)
              printf("  \t Init mat = %.3f \n", seconds() - t0);

            /* U[m]: row-major. in-place update */
            
            /*On GPU*/
            double mttkrp_t0 = seconds();
            MTTKRP_MIHCSR_GPU_oneMode_forCPD(MMCSF, U, Opt, m, it,
              dInds2, dfbrPtr1, dfbrIdx1, dFbrLikeSlcInds, dVals, dU);
            // ((X.ndims == 3) ?  MTTKRP_COO_CPU(X, U, Opt) :  MTTKRP_COO_CPU_4D(X, U, Opt)); 
            // printf("in CPD GPU vals %f\n", U[m].vals[0] );
            // zero_mat(X, U, m);
            // ((X.ndims == 3) ?  MTTKRP_COO_CPU(X, U, Opt) :  MTTKRP_COO_CPU_4D(X, U, Opt)); 
            // printf("after COO vals %f\n", U[m].vals[0] );
            double mttkrp_time = seconds() - mttkrp_t0;
            if(Opt.verbose)
              printf("  \t MTTKRP = %u (%.3f)\n", m+1, mttkrp_time);
            total_mttkrp_time += mttkrp_time;
            
            // fu << "U[m] after MTTKRP: " << endl;
            // write_output(U, m, U_file);

            // Column-major calculation
            if ( m == ndims - 1)
                memcpy(tmpU.vals, U[m].vals, U[m].nRows * rank * sizeof(DTYPE));

            /* Solve ? * tmpATA = U[m] */
            t0 = seconds();
            MatrixSolveNormals(m, ndims, ATA, tmpATA, U[m]);
           if(Opt.verbose)
              printf("  \t MatrixSolveNormals = %.3f \n", seconds() - t0);

            /* Normalized U[m], store the norms in lambda. Use different norms to avoid precision explosion. */
            t0 = seconds();
            if (it == 0 ) {
                Matrix2Norm(U[m], lambda);
            } else {
                MatrixMaxNorm(U[m], lambda);
            }

            cudaMemcpy(dU + loc, &(U[m].vals[0]), U[m].nRows * U[m].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice);
           if(Opt.verbose)
              printf("  \t MatrixNorm = %.3f \n", seconds() - t0);
            // fu << "Normalized U[m]: " << endl;
            // write_output(U, m, U_file);
            // fu << "Init lambda: " << endl;
            // for(ITYPE i = 0; i < rank; ++i)
            //     fu << lambda[i] << "\t" ;
            // fu << endl; 

            /* ATA[m] = U[m]^T * U[m]) */
            t0 = seconds();
            int blas_nrows = (int)(U[m].nRows);
            // cblas_ssyrk(CblasRowMajor, cuplo, ctrans, 
            //     blas_rank, blas_nrows, 
            //     alpha, U[m].vals, blas_rank, 
            //     beta, ATA[m].vals, blas_rank);
            ssyrk_(&uplo, &notrans, 
                  &blas_rank, &blas_nrows, 
                  &alpha, U[m].vals, &blas_rank, 
                  &beta, ATA[m].vals, &blas_rank);
           if(Opt.verbose)
              printf("  \t ssyrk = %.3f \n", seconds() - t0);
            // fa << "Updated ATA[m]: " << endl;
            // write_output(ATA, m, ata_file);

        } // Loop ndims

        // fu << "lambda before KruskalTensorFit: " << endl;
        // for(ITYPE i = 0; i < rank; ++i)
        //     fu << lambda[i] << "\t" ;
        // fu << endl; 
        fit = KruskalTensorFit(X, lambda, U, tmpU, ATA, tmpATA, X_normsq);
        double cpd_time = seconds() - cpd_t0;
        if(it == niters - 1)
          printf("  its = %u (%.3f), fit = %0.5lf  delta = %+0.4e\n",
          it+1, cpd_time, fit, fit - oldfit);

        if(it > 0 && fabs(fit - oldfit) < tol) {
            break;
        }
        oldfit = fit;

    } // Loop niters

    printf("\tTotal MTTKRP time: %.3f sec \n", total_mttkrp_time);
    printf("CPD time (the same measure with SPLATT) = %.3f \n", seconds() - t_cpd);

    GetFinalLambda(rank, ndims, U, lambda);

    // fu.close();
    // fa.close();
    // for(ITYPE m=0; m < ndims; ++m) {
    //   sptFreeMatrix(ATA[m]);
    // }
    delete ATA;

    return fit;

}