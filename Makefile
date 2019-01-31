CXX=g++
# CXX=icc
NVCC=nvcc
NVCC_LIB_PATH=/usr/lib/x86_64-linux-gnu
BOOSTFLAG=-L/usr/local/boost/gnu/7.3/1.67.0/lib -I/usr/local/boost/gnu/7.3/1.67.0/include 
CXXFLAGS=-O3 -std=c++17 -g -fopenmp $(BOOSTFLAG)
 # -fno-tree-vectorize
# -fno-strict-aliasing
# -DUSE_RESTRICT
#NVCCFLAGS +=-O3 -fopenmp -w -restrict #--ptxas-options=-v

# NVCCFLAGS += -O3 -w -gencode arch=compute_60,code=sm_60 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp --std=c++11 -m64 -lineinfo $(BOOSTFLAG) #–default-stream #per-thread #-g #-G
# NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart
NVCCFLAGS += -O3 -w -gencode arch=compute_70,code=sm_70 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp --std=c++11 -m64 -lineinfo $(BOOSTFLAG) #–default-stream #per-thread #-g #-G
NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart
# nvcc -ccbin=/cm/shared/apps/intel/compilers_and_libraries_2016.3.210/linux/bin/intel64/icc -std=c++11 -o t912 t912.cu
all: mttkrp2 ttm

ttm: ttm.cu ttm_cpu.o ttm_gpu.o 
	${NVCC} ${NVCCFLAGS} -o ttm ttm_cpu.o ttm_gpu.o ttm.cu $(NVCCLINKFLAGS)  

mttkrp2: mttkrp.cu mttkrp_gpu.h mttkrp_cpu.h mttkrp_cpu.o 
	${NVCC} ${NVCCFLAGS} -o mttkrp2 mttkrp_cpu.o mttkrp.cu $(NVCCLINKFLAGS)  

ttm_cpu.o: ttm_cpu.h ttm_cpu.cpp util.h
	${CXX} ${CXXFLAGS} -c -o ttm_cpu.o ttm_cpu.cpp

ttm_gpu.o: ttm_gpu.h ttm_gpu.cu util.h
	${NVCC} ${NVCCFLAGS} -c -o ttm_gpu.o ttm_gpu.cu $(NVCCLINKFLAGS)  

mttkrp_cpu.o: mttkrp_cpu.h mttkrp_cpu.cpp util.h
	${CXX} ${CXXFLAGS} -c -o mttkrp_cpu.o mttkrp_cpu.cpp

clean:
	rm -rf mttkrp2 ttm *.o f

# ${NVCC} ${NVCCFLAGS} -ccbin=/opt/intel/compilers_and_libraries_2016/linux/bin/intel64/icc -o mttkrp mttkrp_cpu.o mttkrp.cu $(NVCCLINKFLAGS)  
