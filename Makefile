CXX=g++
# CXX=icc
NVCC=nvcc
NVCC_LIB_PATH=/usr/lib/x86_64-linux-gnu

CXXFLAGS=-O3 -std=c++11 -g -fopenmp
 # -fno-tree-vectorize
# -fno-strict-aliasing
# -DUSE_RESTRICT
#NVCCFLAGS +=-O3 -fopenmp -w -restrict #--ptxas-options=-v

NVCCFLAGS += -O3 -w -arch=sm_37 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp --std=c++11 -m64 -lineinfo #–default-stream #per-thread #-g #-G
NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart
# nvcc -ccbin=/cm/shared/apps/intel/compilers_and_libraries_2016.3.210/linux/bin/intel64/icc -std=c++11 -o t912 t912.cu
all: mttkrp

mttkrp: mttkrp.cu mttkrp_gpu.h mttkrp_cpu.h mttkrp_cpu.o 
	${NVCC} ${NVCCFLAGS} -o mttkrp mttkrp_cpu.o mttkrp.cu $(NVCCLINKFLAGS)  

mttkrp_cpu.o: mttkrp_cpu.h mttkrp_cpu.cpp util.h
	${CXX} ${CXXFLAGS} -c -o mttkrp_cpu.o mttkrp_cpu.cpp

clean:
	rm -rf mttkrp *.o f

# ${NVCC} ${NVCCFLAGS} -ccbin=/opt/intel/compilers_and_libraries_2016/linux/bin/intel64/icc -o mttkrp mttkrp_cpu.o mttkrp.cu $(NVCCLINKFLAGS)  
