## Tensor format

Expects sorted input according to output mode for now. A python script is avaiable to perform the sorting.
example.tns  
3  
3 3 3  
1 1 1 1.00  
1 2 2 2.00
1 3 1 10.00  
2 1 3 7.00  
2 3 1 6.00  
2 3 2 5.00  
3 1 3 3.00  
3 2 2 11.00   

## Build 

$ make  


## Run

To see all the options: 

$ ./mttkrp --help

Example:

$ ./mttkrp -i example.tns -m 0 -R 32 -t 1  

-t 1: COO on CPU  
-t 2: CSF on CPU  
-t 3: COO on GPU  
...

##Notes:

ICC compiler:
Before collecting CPU performance data using icc on P100:
1. copy Makefile_icc to Makefile 
2. in mttkrp.cu file change MTTKRP_HCSR_CPU(X, U, Opt.mode, Opt.R) to MTTKRP_HCSR_CPU_ASR(X, U, Opt.mode, Opt.R);

PAPI:
OSC has PAPI istalled. Copy/edit mttkrp_papi.cu to mttkrp.cu

