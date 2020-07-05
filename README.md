MM-CSF or MixedMode-CSF is a storage-efficient representation for sparse tensors that enables high-performance, compressed and load-balanced execution of tensor kernels on GPUs. Currently, it supports MTTKRP kernel from CP decomposition. In future, we plan to extend it to support all generic sparse tensor kernels. Details can be found in the following links: 
Paper:https://dl.acm.org/doi/abs/10.1145/3295500.3356216 
Slides:http://sc19.supercomputing.org/proceedings/tech_paper/tech_paper_files/pap513s5.pdf

## Tensor format

The input format is expected to start with the number of dimension of the tensor followed by the length of each dimension in the next line. The following lines will have the coordinates and values of each nonzero elements.

An example of a 3x3x3 tensor - toy.tns: 
```
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
```

## Build requirements:
- GCC Compiler 
- CUDA SDK
- Boost C++
- OpenMP



## Build 

`$ make`  

## Run

Example:

`$ ./mttkrp -i toy.tns -m 0 -R 32 -t 1 -f 128`

To see all the options: 
  
`./mttkrp --help`    
```
options:   
        -R rank/feature : set the rank (default 32)  
        -m mode : set the mode of MTTKRP (default 0)  
        -t implementation type: 1: COO CPU, 2: HCSR CPU, 3: COO GPU 4: HCSR GPU 8: B-CSF 10: HB-CSF (default 1)   
        -f fiber-splitting threshold: set the maximum length (nnz) for each fiber. Longer fibers will be split (default inf)  
        -w warp per slice: set number of WARPs assign to per slice  (default 4)  
        -i output file name: e.g., ../dataset/delicious.tns   
        -o output file name: if not set not output file will be written
        


