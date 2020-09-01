#!/bin/bash
#SBATCH --time=5:59:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out
#SBATCH -p batch-bdw-v100


module load cuda
module load mvapich2-2.1/gcc 
path=/home/nisa/Tensor_operations/dataset
bin=/home/nisa/Tensor_operations/SpMTTKRP_GPU/mttkrp
out=/home/nisa/Tensor_operations/SpMTTKRP_GPU/BCSF_ONEBCSF_MMCSF.txt


echo "MM-CSF" >> $out

for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do

    mode=0
    $bin -i $path/$dataset -m $mode -R 32 -t 12 -f 128 -w 1 -s 4 -b 128 -v 1
 	$bin -i $path/$dataset -m $mode -R 32 -t 12 -f 128 -w 1 -s 16 -b 512 -v 1

done


echo "B-CSF" >> $out

for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do
	
	for mode in 0 1 2; do
		# log1=`$bin -i $dataset -m $mode -R 32 -t 8 | perl -p -e 's/\n//'`
		log1=`$bin -i $path/$dataset -m $mode -R 32 -t 8 -f 128 | perl -p -e 's/\n//'`
		echo "$dataset,$log1" >> $out
	done
done

echo "ONE-B-CSF" >> $out
# # # ONE-CSF with most compressed mode
for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do

	if [[ $dataset == "delicious-3d_sorted012.tns" || $dataset == "flickr-3d_sorted012.tns" || $dataset == "1998DARPA_sorted012.tns" ]]; then
		mode=0
	fi
	if [[ $dataset == "nell-1_sorted012.tns" || $dataset == "nell-2_sorted012.tns" ]]; then
		mode=1
	fi
	if [[ $dataset == "freebase_music_sorted012.tns" || $dataset == "freebase_sampled_sorted012.tns" ]]; then
		mode=2
	fi

	log1=`$bin -i $path/$dataset -m $mode -R 32 -t 9 -f 128 | perl -p -e 's/\n//'`
	echo "$dataset,$log1" >> $out

done
