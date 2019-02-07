#!/bin/bash

#PBS -l nodes=1:ppn=28:gpus=1
#PBS -l walltime=5:59:59

module load cuda
path=/users/PAS0134/osu1600/Tensor/dataset
bin=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/mttkrp2
out=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/Loop.txt
echo "MUST fix EXEC!" >> $out
nvprof="nvprof --metrics flop_count_sp,atomic_transactions,dram_read_transactions,dram_write_transactions"

# # mode 0
# nell-1_sorted012.tns nell-2_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns
echo "mode 0" >> $out
for dataset in delicious-3d_sorted012.tns  flickr-3d_sorted012.tns  1998DARPA_sorted012.tns; do
	
	echo "$dataset" >> $out
	mode=0
	# for (( mode = 0; mode < 1; mode++ )); do

		echo "$mode" >> $out

		for blck in 512 256 128 64 32; do		
			for grd in 32768 16384 49152 65536	81920	98304	131072	196608	262144	393216	524288	1048576 ; do				
				log1=`$bin -i $path/$dataset -m $mode -R 32 -t 14 -f 128 -b $blck -g $grd | perl -p -e 's/\n//'`
				echo "$blck,$grd,$log1" >> $out
			done
		done
	# done
done

# # mode 0
# freebase_music_sorted012.tns freebase_sampled_sorted012.tns
echo "mode 1" >> $out
for dataset in nell-1_sorted012.tns nell-2_sorted012.tns ; do
	
	echo "$dataset" >> $out
	mode=1
	# for (( mode = 0; mode < 1; mode++ )); do

		echo "$mode" >> $out

		for blck in 512 256 128 64 32; do		
			for grd in 32768 16384 49152 65536	81920	98304	131072	196608	262144	393216	524288	1048576 ; do				
				log1=`$bin -i $path/$dataset -m $mode -R 32 -t 14 -f 128 -b $blck -g $grd | perl -p -e 's/\n//'`
				echo "$blck,$grd,$log1" >> $out
			done
		done
	# done
done

# # mode 0
# 
echo "mode 2" >> $out
for dataset in freebase_music_sorted012.tns freebase_sampled_sorted012.tns ; do
	
	echo "$dataset" >> $out
	mode=2
	# for (( mode = 0; mode < 1; mode++ )); do

		echo "$mode" >> $out

		for blck in 512 256 128 64 32; do		
			for grd in 32768 16384 49152 65536	81920	98304	131072	196608	262144	393216	524288	1048576 ; do				
				log1=`$bin -i $path/$dataset -m $mode -R 32 -t 14 -f 128 -b $blck -g $grd | perl -p -e 's/\n//'`
				echo "$blck,$grd,$log1" >> $out
			done
		done
	# done
done

		# blck=256
		# for wrp in 1 2 4 8; do				
		# 	log1=`$bin -i $path/$dataset -m $mode -R 32 -t 14 -f 128 -b $blck -w $wrp | perl -p -e 's/\n//'`
		# 	echo "$blck,$wrp,$log1" >> $out
		# done

		# blck=128
		# for wrp in 1 2 4; do				
		# 	log1=`$bin -i $path/$dataset -m $mode -R 32 -t 14 -f 128 -b $blck -w $wrp | perl -p -e 's/\n//'`
		# 	echo "$blck,$wrp,$log1" >> $out
		# done

		# blck=64
		# for wrp in 1 2; do				
		# 	log1=`$bin -i $path/$dataset -m $mode -R 32 -t 14 -f 128 -b $blck -w $wrp | perl -p -e 's/\n//'`
		# 	echo "$blck,$wrp,$log1" >> $out
		# done

		# blck=32
		# for wrp in 1; do				
		# 	log1=`$bin -i $path/$dataset -m $mode -R 32 -t 14 -f 128 -b $blck -w $wrp | perl -p -e 's/\n//'`
		# 	echo "$blck,$wrp,$log1" >> $out
		# done
	



# echo "mode 0, 1, 2" >> $out
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do

# 	$nvprof $bin -i $path/$dataset -m 0 -R 32 -t 8 -f 128 &> tmp
# 	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
# 	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
# 	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
# 	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`

# 	echo "$dataset,$flops,$atomics,$dramR,$dramW" >> $out
# done


# echo "mode 0, 1, 2" >> $out
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do

# 	$nvprof $bin -i $path/$dataset -m 1 -R 32 -t 8 -f 128 &> tmp
# 	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
# 	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
# 	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
# 	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`

# 	echo "$dataset,$flops,$atomics,$dramR,$dramW" >> $out
# done

# echo "mode 0, 1, 2" >> $out
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do

# 	$nvprof $bin -i $path/$dataset -m 2 -R 32 -t 8 -f 128 &> tmp
# 	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
# 	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
# 	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
# 	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`

# 	echo "$dataset,$flops,$atomics,$dramR,$dramW" >> $out
# done



# echo "mode 0, 1, 2" >> $out
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do

# 	$nvprof $bin -i $path/$dataset -m 0 -R 32 -t 14 -f 128  &> tmp
# 	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
# 	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
# 	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
# 	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`

# 	echo "$dataset,$flops,$atomics,$dramR,$dramW" >> $out
# done



# echo "mode 0, 1, 2" >> $out
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do

# 	$nvprof $bin -i $path/$dataset -m 1 -R 32 -t 14 -f 128  &> tmp
# 	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
# 	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
# 	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
# 	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`

# 	echo "$dataset,$flops,$atomics,$dramR,$dramW" >> $out
# done


# echo "mode 0, 1, 2" >> $out
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do

# 	$nvprof $bin -i $path/$dataset -m 2 -R 32 -t 14 -f 128  &> tmp
# 	flops=`grep 'flop_count_sp' tmp | perl -p -e 's/\n//'`
# 	atomics=`grep 'atomic_transactions' tmp | perl -p -e 's/\n//'`
# 	dramR=`grep 'dram_read_transactions' tmp | perl -p -e 's/\n//'`
# 	dramW=`grep 'dram_write_transactions' tmp | perl -p -e 's/\n//'`

# 	echo "$dataset,$flops,$atomics,$dramR,$dramW" >> $out
# done
# echo "mode 0, 1, 2 fiber th" >> $out
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do

# 	# log1=`$bin -i $path/$dataset -m 0 -R 32 -t 8 -f 128 | perl -p -e 's/\n//'`
# 	# log2=`$bin -i $path/$dataset -m 1 -R 32 -t 8 -f 128 | perl -p -e 's/\n//'`
# 	# log3=`$bin -i $path/$dataset -m 2 -R 32 -t 8 -f 128 | perl -p -e 's/\n//'`
# 	# log4=`$bin -i $path/$dataset -m 0 -R 32 -t 12 -f 128 | perl -p -e 's/\n//'`
# 	# log5=`$bin -i $path/$dataset -m 0 -R 32 -t 14  -f 128 | perl -p -e 's/\n//'`
# 	# log6=`$bin -i $path/$dataset -m 1 -R 32 -t 14 -f 128 | perl -p -e 's/\n//'`
# 	# log7=`$bin -i $path/$dataset -m 2 -R 32 -t 14 -f 128 | perl -p -e 's/\n//'`
# 	echo "$dataset,$log1,$log2,$log3,$log4,$log5,$log6,$log7,$log8" >> $out
# done

# # # # mode 1
# echo "mode 1" >> $out
# for dataset in delicious-3d_sorted120.tns nell-1_sorted120.tns nell-2_sorted120.tns flickr-3d_sorted120.tns freebase_music_sorted120.tns freebase_sampled_sorted120.tns 1998DARPA_sorted120.tns brainq_fixed_sorted120.tns ; do
# 	log1=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 -f 16 | perl -p -e 's/\n//'`
# 	log2=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 -f 32 | perl -p -e 's/\n//'`
# 	log3=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 -f 64 | perl -p -e 's/\n//'`
# 	log4=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 -f 128 | perl -p -e 's/\n//'`
# 	log5=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 | perl -p -e 's/\n//'`
# 	echo "$dataset,$log1,$log2,$log3,$log4,$log5" >> $out
# done

# # # # mode 2
# echo "mode 2" >> $out
# for dataset in delicious-3d_sorted201.tns nell-1_sorted201.tns nell-2_sorted201.tns flickr-3d_sorted201.tns freebase_music_sorted201.tns freebase_sampled_sorted201.tns 1998DARPA_sorted201.tns brainq_fixed_sorted201.tns; do
# 	log1=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 -f 16 | perl -p -e 's/\n//'`
# 	log2=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 -f 32 | perl -p -e 's/\n//'`
# 	log3=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 -f 64 | perl -p -e 's/\n//'`
# 	log4=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 -f 128 | perl -p -e 's/\n//'`
# 	log5=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 | perl -p -e 's/\n//'`
# 	echo "$dataset,$log1,$log2,$log3,$log4,$log5" >> $out
# done

# # rm tmp && make && ./mttkrp -i ../dataset/3d_3_8_sorted012.tns -m 0 -R 4 -w 2 -t 8 -o tmp && cat tmp

