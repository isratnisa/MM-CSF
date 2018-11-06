CXX=g++
# CXX=icc
NVCC=nvcc
NVCC_LIB_PATH=/usr/lib/x86_64-linux-gnu

CXXFLAGS=-O3 -std=c++11 -g -fopenmp
 # -fno-tree-vectorize
# -fno-strict-aliasing
# -DUSE_RESTRICT
#NVCCFLAGS +=-O3 -fopenmp -w -restrict #--ptxas-options=-v

NVCCFLAGS += -O3 -w -gencode arch=compute_37,code=sm_37 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp --std=c++11 -m64 -lineinfo #–default-stream #per-thread #-g #-G
NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart
# nvcc -ccbin=/cm/shared/apps/intel/compilers_and_libraries_2016.3.210/linux/bin/intel64/icc -std=c++11 -o t912 t912.cu
all: mttkrp ttm

ttm: ttm.cu ttm_cpu.o ttm_gpu.o 
	${NVCC} ${NVCCFLAGS} -o ttm ttm_cpu.o ttm_gpu.o ttm.cu $(NVCCLINKFLAGS)  

mttkrp: mttkrp.cu mttkrp_gpu.h mttkrp_cpu.h mttkrp_cpu.o 
	${NVCC} ${NVCCFLAGS} -o mttkrp mttkrp_cpu.o mttkrp.cu $(NVCCLINKFLAGS)  

ttm_cpu.o: ttm_cpu.h ttm_cpu.cpp util.h
	${CXX} ${CXXFLAGS} -c -o ttm_cpu.o ttm_cpu.cpp

ttm_gpu.o: ttm_gpu.h ttm_gpu.cu util.h
	${NVCC} ${NVCCFLAGS} -c -o ttm_gpu.o ttm_gpu.cu $(NVCCLINKFLAGS)  

mttkrp_cpu.o: mttkrp_cpu.h mttkrp_cpu.cpp util.h
	${CXX} ${CXXFLAGS} -c -o mttkrp_cpu.o mttkrp_cpu.cpp

clean:
	rm -rf mttkrp ttm *.o f

# ${NVCC} ${NVCCFLAGS} -ccbin=/opt/intel/compilers_and_libraries_2016/linux/bin/intel64/icc -o mttkrp mttkrp_cpu.o mttkrp.cu $(NVCCLINKFLAGS)  
