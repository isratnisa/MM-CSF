#!/bin/bash

#PBS -l nodes=1:ppn=28:gpus=1
#PBS -l walltime=5:59:59

module load cuda
path=/users/PAS0134/osu1600/Tensor/dataset
bin=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/mttkrp
out=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/dense_tns_nvprofl2data.txt
nvprof="nvprof --metrics global_atomic_requests,flop_count_sp,atomic_transactions,l2_tex_read_hit_rate,l2_tex_write_hit_rate,dram_read_transactions,dram_write_transactions,sm_efficiency,achieved_occupancy"


# #MI-CSF
# # for dataset in $path/syn_mtx/*.tns; do
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns  nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do

# 	$nvprof $bin -i $path/$dataset -m 0 -R 32 -t 12 -f 128 -h 5 &> tmp
# 	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
# 	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
# 	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
# 	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`
# 	smEff=`grep 'sm_efficiency' tmp | perl -p -e 's/\n//'`
# 	achOcc=`grep 'achieved_occupancy' tmp | perl -p -e 's/\n//'`

# 	echo "$dataset,$flops,$atomics,$dramR,$dramW,$smEff,$achOcc" >> $out

# done

#ALL-CSF
for dataset in $path/dense_syn_tns/*.tns; do
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns  nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do

	for mode in 0 1 2; do

		$nvprof $bin -i $dataset -m $mode -R 32 -t 8 &> tmp
		# $nvprof $bin -i $path/$dataset -m $mode -R 32 -t 8 -f 128 &> tmp
		flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
		atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
		dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
		dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`
		l2Read=`grep 'l2_tex_read_hit_rate' tmp | perl -p -e 's/\n//'`
		l2Write=`grep 'l2_tex_write_hit_rate' tmp | perl -p -e 's/\n//'`
		smEff=`grep 'sm_efficiency' tmp | perl -p -e 's/\n//'`
		achOcc=`grep 'achieved_occupancy' tmp | perl -p -e 's/\n//'`

		echo "$dataset,$flops,$atomics,$dramR,$dramW,$smEff,$achOcc,$l2Read,$l2Write" >> $out
	done
done

# # ONE-CSF with most compressed mode
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do

# 	if [[ $dataset == "delicious-3d_sorted012.tns" || $dataset == "flickr-3d_sorted012.tns" || $dataset == "1998DARPA_sorted012.tns" ]]; then
# 		mode=0
# 	fi
# 	if [[ $dataset == "nell-1_sorted012.tns" || $dataset == "nell-2_sorted012.tns" ]]; then
# 		mode=1
# 	fi
# 	if [[ $dataset == "freebase_music_sorted012.tns" || $dataset == "freebase_sampled_sorted012.tns" ]]; then
# 		mode=2
# 	fi
for dataset in $path/dense_syn_tns/*.tns; do

	$nvprof $bin -i $dataset -m 0 -R 32 -t 14 &> tmp

	# $nvprof $bin -i $path/$dataset -m $mode -R 32 -t 14 -f 128 &> tmp
	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`
	l2Read=`grep 'l2_tex_read_hit_rate' tmp | perl -p -e 's/\n//'`
	l2Write=`grep 'l2_tex_write_hit_rate' tmp | perl -p -e 's/\n//'`
	smEff=`grep 'sm_efficiency' tmp | perl -p -e 's/\n//'`
	achOcc=`grep 'achieved_occupancy' tmp | perl -p -e 's/\n//'`

	echo "$dataset,$flops,$atomics,$dramR,$dramW,$smEff,$achOcc,$l2Read,$l2Write" >> $out
done





