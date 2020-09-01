MM-CSF or MixedMode-CSF is a CSF based storage format that partitions the tensor's nonzero elements into disjoint sections, each of which is compressed to create fibers along a different mode. It enables high-performance, compressed, and load-balanced execution of tensor kernels on GPUs. Currently, it supports MTTKRP kernel from CP decomposition. In future, we plan to extend it to support all generic sparse tensor kernels. This is a followup work of BCSF (https://ieeexplore.ieee.org/document/8821030), published in IPDPS'2019. Details of MM-CSF be found in the following links:  
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
- LAPACK


## Build
Set LAPACK\_HOME path in the Makefile.   
`$ cd src && make`  

## Run

Example:

1.mttkrp using COO format on CPU:  
`$ ./src/mttkrp -i toy.tns -m 0 -R 32 -t 1 -f 128`  

2.mttkrp using BCSF format on GPU:  
`$ ./src/mttkrp -i toy.tns -m 0 -R 32 -t 8 -f 128`  

3.mttkrp using MM-CSF format on GPU:  
`$ ./src/mttkrp -i toy.tns -m 0 -R 32 -t 12 -f 128 -w 1`  

More examples can be found in the scripts folder.

To see all the options: 
  
`./mttkrp --help`    
```
options:   
        -R rank/feature : set the rank (default 32)  
        -m mode : set the mode of MTTKRP (default 0, MMCSF evaluates all modes)  
        -v verbose: set to 1 to enable
        -t implementation type: 1: COO CPU, 3: COO GPU 8: B-CSF 10: HB-CSF 12: MM-CSF on GPU (default 1)   
        -f fiber-splitting threshold: set the maximum length (nnz) for each fiber. Longer fibers will be split (default inf)  
        -w warp per slice: set number of WARPs assign to per slice  (default 4)  
        -i intput file name: e.g., ../dataset/delicious.tns   
        -o output file name: if not set not output file will be written
        


