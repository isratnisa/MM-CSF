#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out
#SBATCH -p batch-gpu

module load cuda
path=../dataset
bin=/home/nisa/Tensor_operations/MTTKRP/mttkrp
out=Tiled_bin.txt

# mode 0

echo "mode 0" >> $out
for dataset in delicious-3d_sorted012.tns nell-1_sorted012.tns nell-2_sorted012.tns flickr-3d_sorted012.tns freebase_music_sorted012.tns freebase_sampled_sorted012.tns 1998DARPA_sorted012.tns brainq_fixed_sorted012.tns; do
	log1=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 8 -f 16 -l 2 | perl -p -e 's/\n//'`
	log2=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 8 -f 32 -l 2 | perl -p -e 's/\n//'`
	log3=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 8 -f 64 -l 2 | perl -p -e 's/\n//'`
	log4=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 8 -f 128 -l 2| perl -p -e 's/\n//'`
	log5=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 8 -l 2 | perl -p -e 's/\n//'`
	log6=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 8 -l 1 | perl -p -e 's/\n//'`
	log7=`$bin -i $path/$dataset -m 0 -R 32 -w 4 -t 8 -f 128 -l 1 | perl -p -e 's/\n//'`
	echo "$dataset,$log1,$log2,$log3,$log4,$log5,$log6,$log7" >> $out
done

# # # mode 1
echo "mode 1" >> $out
for dataset in delicious-3d_sorted120.tns nell-1_sorted120.tns nell-2_sorted120.tns flickr-3d_sorted120.tns freebase_music_sorted120.tns freebase_sampled_sorted120.tns 1998DARPA_sorted120.tns brainq_fixed_sorted120.tns ; do
	log1=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 16 -l 2  | perl -p -e 's/\n//'`
	log2=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 32 -l 2 |  perl -p -e 's/\n//'`
	log3=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 64 -l 2 |  perl -p -e 's/\n//'`
	log4=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 128 -l 2 |  perl -p -e 's/\n//'`
	log5=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -l 2 |  perl -p -e 's/\n//'`
	log6=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -l 1 |  perl -p -e 's/\n//'`
	log7=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 128 -l 1 | perl -p -e 's/\n//'`
	echo "$dataset,$log1,$log2,$log3,$log4,$log5,$log6,$log7" >> $out
done

# # # mode 2
echo "mode 2" >> $out
for dataset in delicious-3d_sorted201.tns nell-1_sorted201.tns nell-2_sorted201.tns flickr-3d_sorted201.tns freebase_music_sorted201.tns freebase_sampled_sorted201.tns 1998DARPA_sorted201.tns brainq_fixed_sorted201.tns; do
	log1=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 16 -l 2 | perl -p -e 's/\n//'`
	log2=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 32 -l 2 |  perl -p -e 's/\n//'`
	log3=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 64 -l 2 |  perl -p -e 's/\n//'`
	log4=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 128 -l 2 |  perl -p -e 's/\n//'`
	log5=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -l 2 |  perl -p -e 's/\n//'`
	log6=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -l 1 |  perl -p -e 's/\n//'`
	log7=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 128 -l 1 | perl -p -e 's/\n//'`
	echo "$dataset,$log1,$log2,$log3,$log4,$log5,$log6,$log7" >> $out
done

# # rm tmp && make && ./mttkrp -i ../dataset/3d_3_8_sorted012.tns -m 0 -R 4 -w 2 -t 8 -o tmp && cat tmp

