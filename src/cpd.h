#ifndef CPD_H
#define CPD_H

#include "util.h"
#include "mttkrp_cpu.h"

double cpd(Tensor &X, Matrix * U, Options Opt, DTYPE * lambda);

double cpd_new(Tensor &X, Matrix * U, Options Opt, DTYPE * lambda);

int getNumThread();
#endif