#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out
#SBATCH -p batch-gpu

module load cuda
path=../dataset
bin=/home/nisa/Tensor_operations/SpMTTKRP/mttkrp
out=/home/nisa/Tensor_operations/SpMTTKRP/4d_sorted.txt

# mode 0

echo "mode 0" >> $out
for dataset in nips_sorted0123.tns uber_sorted0123.tns ; do
	
	for mode in 0 1 2 3; do
		log1=`$bin -i $path/$dataset -m $mode -R 32 -t 12 -f 128 -c 1 `
		echo "$dataset,$mode,$log1"
	done
done

