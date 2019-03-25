#ifndef MTTKRP_GPU_H
#define MTTKRP_GPU_H

#include "util.h"

int MTTKRP_COO_GPU(const Tensor &X, Matrix *U, const Options Opt);

int MTTKRP_HCSR_GPU(Tensor &X, Matrix *U, const Options &Opt);

int MTTKRP_TILED_COO_GPU(TiledTensor *TiledX, Matrix *U, const Options Opt);

int MTTKRP_B_HCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

int MTTKRP_ONE_HCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

int MTTKRP_HYB_GPU(const HYBTensor &HybX, Matrix *U, const Options &Opt);

int MTTKRP_MIHCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

int MTTKRP_MIHCSR_multiGPU(TiledTensor *TiledX, Matrix *U, const Options &Opt, const MPI_param &MPIparam);

#endif