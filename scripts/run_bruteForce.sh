#!/bin/bash

#PBS -l nodes=1:ppn=28:gpus=1
#PBS -l walltime=11:59:59

module load cuda
path=/users/PAS0134/osu1600/Tensor/dataset
bin=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/mttkrp
out=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/bruteFrce

p0="012"
p1="021"
p2="120"
p3="102"
p4="201"
p5="210"

# delicious-3d_sorted012.tns
for dataset in  nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do
# for dataset in 3d_3_8_sorted012.tns; do

	out=$out$dataset.txt

	echo "$dataset" >> $out

	for perm0 in $p0 $p1 $p2 $p3 $p4 $p5; do
		
		for perm1 in $p0 $p1 $p2 $p3 $p4 $p5; do
		
			for perm2 in $p0 $p1 $p2 $p3 $p4 $p5; do
				
					
					log1=`$bin -i $path/$dataset -m 0 -R 32 -t 12 -f 128 -p $perm0 -q $perm1 -r $perm2`

					echo "$log1" >> $out
			done
		done
	done
done

