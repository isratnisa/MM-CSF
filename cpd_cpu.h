#ifndef CPD_CPU_H
#define CPD_CPU_H

#include "util.h"
#include "mttkrp_cpu.h"
#include "mttkrp_gpu.h"

// double cpd(Tensor &X, Matrix * U, Options Opt, DTYPE * lambda);
double cpd(Tensor &X, TiledTensor *MMCSF, Matrix * U, Options Opt, DTYPE * lambda);

double cpd_new(Tensor &X, Matrix * U, Options Opt, DTYPE * lambda);


#endif