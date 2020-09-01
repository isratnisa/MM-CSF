#ifndef MTTKRP_GPU_H
#define MTTKRP_GPU_H

#include "util.h"

int MTTKRP_COO_GPU(const Tensor &X, Matrix *U, const Options Opt);

int MTTKRP_HCSR_GPU(Tensor &X, Matrix *U, const Options &Opt);

int MTTKRP_TILED_COO_GPU(TiledTensor *TiledX, Matrix *U, const Options Opt);

int MTTKRP_B_HCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

int MTTKRP_B_HCSR_GPU_ANYMODE(TiledTensor *TiledX, Matrix *U, const Options &Opt, int mode);

int MTTKRP_ONE_HCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

int MTTKRP_HYB_GPU(const HYBTensor &HybX, Matrix *U, const Options &Opt);

int MTTKRP_MIHCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

// int MTTKRP_MIHCSR_GPU_oneMode_forCPD(TiledTensor *TiledX, Matrix *U, const Options &Opt, int cpdMode, int iter);

int init_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt, ITYPE **dInds2, ITYPE **dfbrPtr1, ITYPE **dfbrIdx1, ITYPE **dFbrLikeSlcInds, DTYPE **dVals, DTYPE **dU);

int MTTKRP_MIHCSR_GPU_oneMode_forCPD(TiledTensor *TiledX, Matrix *U, const Options &Opt, int cpdMode, int iter,

	ITYPE *dInds2, ITYPE *dfbrPtr1, ITYPE *dfbrIdx1, ITYPE *dFbrLikeSlcInds, DTYPE *dVals, DTYPE *dU);

#endif