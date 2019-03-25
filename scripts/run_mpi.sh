#!/bin/bash
#PBS -l nodes=2:ppn=14:gpus=1
#PBS -l walltime=5:59:59

module load cuda
nvidia-smi

module load cuda
path=/users/PAS0134/osu1600/Tensor/dataset
bin="mpiexec -n 2 -ppn 1 /users/PAS0134/osu1600/Tensor/SpMTTKRP_GPU/mttkrp"
out=/users/PAS0134/osu1600/Tensor/SpMTTKRP_GPU/mpi.txt

# make && nvprof --profile-child-processes  --metrics all mpiexec -n 1 -ppn 1 /users/PAS0134/osu1601/israt/git/SDDMM_GPU/big_tile_MPI/sddmm ~/israt/graph_data/nips.mtx 32 256 999999 &> nvporf_1node_on2nodemachine.csv

# make && mpiexec -n 2 -ppn 1 $bin ~/israt/graph_data/nytimes.mtx 32 256 9999999
# uber_sorted1230.tns nips_sorted1230.tns chicago-crime-comm_sorted1230.tns flickr-4d_sorted1230.tns enron_sorted1230.tns ; do
	
echo "Starting MM-CSF" >> $out
mode=0
for dataset in 3d_3_8_sorted012.tns; do
	log1=`$bin -i $path/$dataset -m $mode -R 32 -t 12 -f 128 -b 128 -w 1 -s 1 | perl -p -e 's/\n//'`
	echo "$dataset,$log1" >> $out

done
