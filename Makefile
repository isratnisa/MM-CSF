CXX=g++
# CXX=icc
NVCC=nvcc
NVCC_LIB_PATH=/usr/lib/x86_64-linux-gnu
# pitzer
BOOSTFLAG=-L/apps/boost/intel/18.0/1.67.0/lib -I/apps/boost/intel/18.0/1.67.0/include 
# BOOSTFLAG=-L/usr/local/boost/gnu/7.3/1.67.0/lib -I/usr/local/boost/gnu/7.3/1.67.0/include

LAPACK_LIB_PATH=/users/PAS0134/osu1600/softwares/OpenBLAS/lib/
LAPACK_INCLUDE_PATH=/users/PAS0134/osu1600/softwares/OpenBLAS/include/
BLAS_INCLUDE_PATH=

CXXFLAGS=-O3 -std=c++11 -g -fopenmp $(BOOSTFLAG) -I${LAPACK_INCLUDE_PATH} -I${BLAS_INCLUDE_PATH} -DADD_ 
CXXLINKFLAGS=-L${LAPACK_LIB_PATH} -lopenblas

MPIFLAGS = -I/opt/mvapich2/intel/18.0/2.3/include
MPILINKFLAGS=-L/opt/mvapich2/intel/18.0/2.3/lib -lmpi

NVCCFLAGS += -O3 -w -gencode arch=compute_70,code=sm_70 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp --std=c++11 -m64 -lineinfo $(BOOSTFLAG) #–default-stream #per-thread #-g #-G
NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart
# nvcc -ccbin=/cm/shared/apps/intel/compilers_and_libraries_2016.3.210/linux/bin/intel64/icc -std=c++11 -o t912 t912.cu
all: mttkrp ttm cpd

cpd: cpd.cu cpd_cpu.o mttkrp_cpu.h clapack.h mttkrp_cpu.o mttkrp_gpu.o 
	${NVCC} ${NVCCFLAGS} -o cpd cpd_cpu.o mttkrp_cpu.o mttkrp_gpu.o cpd.cu $(NVCCLINKFLAGS) ${CXXLINKFLAGS} ${MPIFLAGS} ${MPILINKFLAGS}

ttm: ttm.cu ttm_cpu.o ttm_gpu.o 
	${NVCC} ${NVCCFLAGS} -o ttm ttm_cpu.o ttm_gpu.o ttm.cu $(NVCCLINKFLAGS)  

mttkrp: mttkrp.cu mttkrp_cpu.o mttkrp_gpu.o 
	${NVCC} ${NVCCFLAGS} ${MPIFLAGS} ${MPILINKFLAGS} -o mttkrp mttkrp_cpu.o mttkrp_gpu.o mttkrp.cu $(NVCCLINKFLAGS)

mttkrp_gpu.o: mttkrp_gpu.h mttkrp_gpu.cu util.h mttkrp_mpi.h
	${NVCC} ${NVCCFLAGS} ${MPIFLAGS} ${MPILINKFLAGS} -c -o mttkrp_gpu.o mttkrp_gpu.cu $(NVCCLINKFLAGS)  

mttkrp_cpu.o: mttkrp_cpu.h mttkrp_cpu.cpp util.h
	${CXX} ${CXXFLAGS} -c -o mttkrp_cpu.o mttkrp_cpu.cpp

cpd_cpu.o: cpd_cpu.cpp util.h clapack.h  mttkrp_cpu.h 
	${CXX} ${CXXFLAGS} -c -o cpd_cpu.o cpd_cpu.cpp

ttm_gpu.o: ttm_gpu.h ttm_gpu.cu util.h
	${NVCC} ${NVCCFLAGS} -c -o ttm_gpu.o ttm_gpu.cu $(NVCCLINKFLAGS)  

ttm_cpu.o: ttm_cpu.h ttm_cpu.cpp util.h
	${CXX} ${CXXFLAGS} -c -o ttm_cpu.o ttm_cpu.cpp

clean:
	rm -rf mttkrp ttm cpd *.o f

# ${NVCC} ${NVCCFLAGS} -ccbin=/opt/intel/compilers_and_libraries_2016/linux/bin/intel64/icc -o mttkrp mttkrp_cpu.o mttkrp.cu $(NVCCLINKFLAGS)  
