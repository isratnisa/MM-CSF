#!/bin/bash

#PBS -l nodes=1:ppn=28:gpus=1
#PBS -l walltime=5:59:59

module load cuda
path=/users/PAS0134/osu1600/Tensor/dataset
bin=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/mttkrp
out=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/includingmemcopy.txt

# mode 0

echo "mode 0" >> $out
for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do
	log1=`$bin -i $path/$dataset -m 0 -R 32 -t 9 -f 128 | perl -p -e 's/\n//'`
	log2=`$bin -i $path/$dataset -m 0 -R 32 -t 8 -f 128 | perl -p -e 's/\n//'`
	log3=`$bin -i $path/$dataset -m 0 -R 32 -t 3 | perl -p -e 's/\n//'`
	# log2=`$bin -i $path/$dataset -m 0 -R 32 -t 10 | perl -p -e 's/\n//'`
	echo "$dataset,$log1,$log2,$log3,$log4,$log5" >> $out
done

echo "mode 1" >> $out
for dataset in delicious-3d_sorted120.tns nell-1_sorted120.tns nell-2_sorted120.tns flickr-3d_sorted120.tns freebase_music_sorted120.tns freebase_sampled_sorted120.tns 1998DARPA_sorted120.tns brainq_fixed_sorted120.tns ; do
	log1=`$bin -i $path/$dataset -m 1 -R 32 -t 9 -f 128 | perl -p -e 's/\n//'`
	log2=`$bin -i $path/$dataset -m 1 -R 32 -t 8 -f 128 | perl -p -e 's/\n//'`
	log3=`$bin -i $path/$dataset -m 1 -R 32 -t 3 | perl -p -e 's/\n//'`
	echo "$dataset,$log1,$log2,$log3,$log4,$log5" >> $out
done

echo "mode 2" >> $out
for dataset in delicious-3d_sorted201.tns nell-1_sorted201.tns nell-2_sorted201.tns flickr-3d_sorted201.tns freebase_music_sorted201.tns freebase_sampled_sorted201.tns 1998DARPA_sorted201.tns brainq_fixed_sorted201.tns; do
	log1=`$bin -i $path/$dataset -m 2 -R 32 -t 9 -f 128 | perl -p -e 's/\n//'`
	log2=`$bin -i $path/$dataset -m 2 -R 32 -t 8 -f 128 | perl -p -e 's/\n//'`
	log3=`$bin -i $path/$dataset -m 2 -R 32 -t 3 | perl -p -e 's/\n//'`
	echo "$dataset,$log1,$log2,$log3,$log4,$log5" >> $out
done


# echo "mode 0" >> $out
# for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do
# 	log1=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 9 -f 16 | perl -p -e 's/\n//'`
# 	log2=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 9 -f 32 | perl -p -e 's/\n//'`
# 	log3=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 9 -f 64 | perl -p -e 's/\n//'`
# 	log4=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 9 -f 128 | perl -p -e 's/\n//'`
# 	log5=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 9 | perl -p -e 's/\n//'`
# 	echo "$dataset,$log1,$log2,$log3,$log4,$log5" >> $out
# done

# # # mode 1
# for dataset in delicious-3d_sorted120.tns nell-1_sorted120.tns nell-2_sorted120.tns flickr-3d_sorted120.tns freebase_music_sorted120.tns freebase_sampled_sorted120.tns 1998DARPA_sorted120.tns brainq_fixed_sorted120.tns ; do
# 	log1=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 16 | perl -p -e 's/\n//'`
# 	log2=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 32 | perl -p -e 's/\n//'`
# 	log3=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 64 | perl -p -e 's/\n//'`
# 	log4=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 128 | perl -p -e 's/\n//'`
# 	log5=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 512 | perl -p -e 's/\n//'`
# 	echo "$dataset,$log1,$log2,$log3,$log4,$log5" >> slicBin_mode1.txt
# done

# # # mode 2
# for dataset in delicious-3d_sorted201.tns nell-1_sorted201.tns nell-2_sorted201.tns flickr-3d_sorted201.tns freebase_music_sorted201.tns freebase_sampled_sorted201.tns 1998DARPA_sorted201.tns brainq_fixed_sorted201.tns; do
# 	log1=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 16 | perl -p -e 's/\n//'`
# 	log2=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 32 | perl -p -e 's/\n//'`
# 	log3=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 64 | perl -p -e 's/\n//'`
# 	log4=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 128 | perl -p -e 's/\n//'`
# 	log5=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 512 | perl -p -e 's/\n//'`
# 	echo "$dataset,$log1,$log2,$log3,$log4,$log5" >> slicBin_mode2.txt
# done

# # rm tmp && make && ./mttkrp -i ../dataset/3d_3_8_sorted012.tns -m 0 -R 4 -w 2 -t 8 -o tmp && cat tmp

