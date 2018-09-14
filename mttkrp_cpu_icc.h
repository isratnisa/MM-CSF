#ifndef MTTKRP_CPU_H
#define MTTKRP_CPU_H

#include "util.h"

void Hello();


//implementation 1; MTTKRP on CPU using COO
int MTTKRP_COO_CPU(const Tensor &X, Matrix *U, const Options &Opt);

//implementation 5; Tiled MTTKRP on CPU using COO
int MTTKRP_TILED_COO_CPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

//implementation 2
int MTTKRP_HCSR_CPU(const Tensor &X, Matrix *U, const Options &Opt);

int MTTKRP_HYB_CSL_CPU( HYBTensor &HybX, Matrix *U, Options &Opt);

int MTTKRP_HYB_HCSR_CPU(HYBTensor &X, Matrix *U, Options &Opt);

int MTTKRP_HYB_CPU(HYBTensor &HybX, Matrix *U, Options &Opt, int iter);

//Aravind optimized
int MTTKRP_HCSR_CPU_ASR(const Tensor &X, Matrix *U, const Options &Opt);

int MTTKRP_TILED_HCSR_CPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);
 
#endif
