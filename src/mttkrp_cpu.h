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

#ifndef MTTKRP_CPU_H
#define MTTKRP_CPU_H

#include "util.h"

void Hello();


//implementation 1; MTTKRP on CPU using COO
int MTTKRP_COO_CPU(const Tensor &X, Matrix *U, const Options &Opt);

int MTTKRP_COO_CPU_4D(const Tensor &X, Matrix *U, const Options &Opt);

//implementation 5; Tiled MTTKRP on CPU using COO
int MTTKRP_TILED_COO_CPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

//implementation 2
int MTTKRP_HCSR_CPU(const Tensor &X, const TiledTensor *TiledX, Matrix *U, const Options &Opt);

int MTTKRP_HCSR_CPU_slc(const Tensor &X, const TiledTensor *TiledX, Matrix *U, const Options &Opt);

int MTTKRP_MIHCSR_CPU(const TiledTensor *TiledX, Matrix *U, const Options &Opt, const int mode);

int MTTKRP_HCSR_CPU_mode1(const Tensor &X, Matrix *U, const Options &Opt, const int mode);

int MTTKRP_HCSR_CPU_mode2(const Tensor &X, Matrix *U, const Options &Opt, const int mode);

int MTTKRP_MIHCSR_CPU_FBR_ATOMICS(const TiledTensor *TiledX, Matrix *U, const Options &Opt, const int mode, const int MTTKRPmode);

int MTTKRP_MIHCSR_CPU_ALL_ATOMICS(const TiledTensor *TiledX, Matrix *U, const Options &Opt, const int HCSRmode, const int MTTKRPmode);

int MTTKRP_HCSR_CPU_4D(const Tensor &X, Matrix *U, const Options &Opt);

int MTTKRP_HYB_CSL_CPU( HYBTensor &HybX, Matrix *U, Options &Opt);

int MTTKRP_HYB_COO_CPU_naive_4D(const HYBTensor &HybX, Matrix *U, const Options &Opt);

int MTTKRP_HYB_HCSR_CPU(HYBTensor &X, Matrix *U, Options &Opt);

int MTTKRP_HYB_HCSR_CPU_4D(const HYBTensor &X, Matrix *U, const Options &Opt);

int MTTKRP_HYB_CPU(HYBTensor &HybX, Matrix *U, Options &Opt);

int MTTKRP_HYB_CPU_4D(HYBTensor &HybX, Matrix *U, Options &Opt);

//Aravind optimized
int MTTKRP_HCSR_CPU_ASR(const Tensor &X, Matrix *U, ITYPE mode, ITYPE R);

int MTTKRP_TILED_HCSR_CPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

int MTTKRP_TILED_HCSR_CPU_4D(TiledTensor *TiledX, Matrix *U, const Options &Opt);
 
#endif
