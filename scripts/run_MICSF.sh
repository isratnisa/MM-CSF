#!/bin/bash

#PBS -l nodes=1:ppn=28:gpus=1
#PBS -l walltime=5:59:59

module load cuda
path=/users/PAS0134/osu1600/Tensor/dataset
bin=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/mttkrp
out=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/perf_dense_syn.txt

for dataset in $path/dense_syn_tns/*.tns; do
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns  nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do
	for mode in 0 1 2; do
		log1=`$bin -i $dataset -m $mode -R 32 -t 8 | perl -p -e 's/\n//'`
		echo "$dataset,$mode,$log1" >> $out
	done

		log1=`$bin -i $dataset -m 0 -R 32 -t 14 | perl -p -e 's/\n//'`
		# log1=`$bin -i $path/$dataset -m 0 -R 32 -t 12 -f 128 -h 2 | perl -p -e 's/\n//'`
		# log2=`$bin -i $path/$dataset -m 0 -R 32 -t 12 -f 128 -h 5 | perl -p -e 's/\n//'`
		# log2=`$bin -i $dataset -m 0 -R 32 -t 12 -f 128 -h 2 | perl -p -e 's/\n//'`
		# log3=`$bin -i $dataset -m 0 -R 32 -t 12 -f 128 -h 5 | perl -p -e 's/\n//'`
		# log4=`$bin -i $dataset -m 0 -R 32 -t 12 -f 128 -h 10 | perl -p -e 's/\n//'`

		echo "$dataset,$log1" >> $out
done





