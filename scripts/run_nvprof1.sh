#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out
#SBATCH -p batch-bdw-v100

module load cuda
path=/home/nisa/Tensor_operations/dataset
bin=/home/nisa/Tensor_operations/SpMTTKRP_private/mttkrp
out=/home/nisa/Tensor_operations/SpMTTKRP_private/nvprof_mode0MMslcAtomics_new.txt
echo "check kernel"
# nvprof="nvprof --metrics  inst_executed_global_atomics,flop_count_sp,atomic_transactions,l2_tex_read_hit_rate,l2_tex_write_hit_rate,dram_read_transactions,dram_write_transactions,sm_efficiency,achieved_occupancy"
nvprof="nvprof --metrics dram_read_transactions,l2_tex_read_hit_rate,gld_transactions"


#MI-CSF
# for dataset in $path/syn_mtx/*.tns; do
	# delicious-3d_sorted012.tns
for dataset in  nell-1_sorted012.tns  nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do

	$nvprof $bin -i $path/$dataset -m 0 -R 32 -t 12 -f 128 -h 5 &> tmp
	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`
	smEff=`grep 'sm_efficiency' tmp | perl -p -e 's/\n//'`
	l2Read=`grep 'l2_tex_read_hit_rate' tmp | perl -p -e 's/\n//'`
	l2Write=`grep 'l2_tex_write_hit_rate' tmp | perl -p -e 's/\n//'`
	achOcc=`grep 'achieved_occupancy' tmp | perl -p -e 's/\n//'`
	gld=`grep 'gld_transactions' tmp | perl -p -e 's/\n//'`

	echo "$dataset,$flops,$atomics,$dramR,$dramW,$smEff,$achOcc,$l2Read,$l2Write,$gld" >> $out

done

# #ALL-CSF
# # for dataset in $path/dense_syn_tns/*.tns; do
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns  nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns; do

# 	for mode in 0 1 2; do

# 		# $nvprof $bin -i $dataset -m $mode -R 32 -t 8 &> tmp
# 		$nvprof $bin -i $path/$dataset -m $mode -R 32 -t 8 -f 128 &> tmp
# 		flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
# 		atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
# 		dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
# 		dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`
# 		l2Read=`grep 'l2_tex_read_hit_rate' tmp | perl -p -e 's/\n//'`
# 		l2Write=`grep 'l2_tex_write_hit_rate' tmp | perl -p -e 's/\n//'`
# 		smEff=`grep 'sm_efficiency' tmp | perl -p -e 's/\n//'`
# 		achOcc=`grep 'achieved_occupancy' tmp | perl -p -e 's/\n//'`

# 		echo "$dataset,$flops,$atomics,$dramR,$dramW,$smEff,$achOcc,$l2Read,$l2Write" >> $out
# 	done
# done

# # # ONE-CSF with most compressed mode
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
# # for dataset in $path/dense_syn_tns/*.tns; do

# 	# $nvprof $bin -i $dataset -m 0 -R 32 -t 14 &> tmp

# 	$nvprof $bin -i $path/$dataset -m $mode -R 32 -t 14 -f 128 &> tmp
# 	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
# 	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
# 	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
# 	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`
# 	l2Read=`grep 'l2_tex_read_hit_rate' tmp | perl -p -e 's/\n//'`
# 	l2Write=`grep 'l2_tex_write_hit_rate' tmp | perl -p -e 's/\n//'`
# 	smEff=`grep 'sm_efficiency' tmp | perl -p -e 's/\n//'`
# 	achOcc=`grep 'achieved_occupancy' tmp | perl -p -e 's/\n//'`

# 	echo "$dataset,$flops,$atomics,$dramR,$dramW,$smEff,$achOcc,$l2Read,$l2Write" >> $out
# done





