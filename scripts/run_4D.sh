#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out
#SBATCH -p batch-gpu

module load cuda
path=../dataset
bin=/home/nisa/Tensor_operations/SpMTTKRP_private/mttkrp
out=/home/nisa/Tensor_operations/SpMTTKRP_private/4d_sorted.txt

# mode 0

echo "mode 0" >> $out
for dataset in nips_sorted0123.tns uber_sorted0123.tns ; do
	
	for mode in 0 1 2 3; do
		log1=`$bin -i $path/$dataset -m $mode -R 32 -t 12 -f 128 -c 1 `
		echo "$dataset,$mode,$log1"
	done

	# for mode in 0 1 2 3; do
	# 	log1=`$bin -i $path/$dataset -m $mode -R 32 -t 10 -f 128 -c 1 | perl -p -e 's/\n//'`
	# 	echo "$dataset,$mode,$log1" >> $out
	# done
done

# # # # mode 1
# echo "mode 1" >> $out
# for dataset in uber_sorted1230.tns nips_sorted1230.tns chicago-crime-comm_sorted1230.tns flickr-4d_sorted1230.tns ; do
# 	log1=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 -f 16 | perl -p -e 's/\n//'`
# 	log2=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 -f 32 | perl -p -e 's/\n//'`
# 	log3=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 -f 64 | perl -p -e 's/\n//'`
# 	log4=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 -f 128 | perl -p -e 's/\n//'`
# 	log5=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 9 | perl -p -e 's/\n//'`
# 	log6=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 | perl -p -e 's/\n//'`
# 	log7=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 128 | perl -p -e 's/\n//'`
# 	log8=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -f 128 -l 2| perl -p -e 's/\n//'`
# 	log9=`$bin -i $path/$dataset -m 1 -R 32 -w 4 -t 8 -l 2 | perl -p -e 's/\n//'`
# 	echo "$dataset,$log1,$log2,$log3,$log4,$log5,$log6,$log7,$log8,$log9" >> $out
# done

# # # # mode 2
# echo "mode 2" >> $out
# for dataset in uber_sorted2301.tns nips_sorted2301.tns chicago-crime-comm_sorted2301.tns flickr-4d_sorted2301.tns; do
# 	log1=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 -f 16 | perl -p -e 's/\n//'`
# 	log2=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 -f 32 | perl -p -e 's/\n//'`
# 	log3=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 -f 64 | perl -p -e 's/\n//'`
# 	log4=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 -f 128 | perl -p -e 's/\n//'`
# 	log5=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 9 | perl -p -e 's/\n//'`
# 	log6=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 | perl -p -e 's/\n//'`
# 	log7=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 128 | perl -p -e 's/\n//'`
# 	log8=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -f 128 -l 2| perl -p -e 's/\n//'`
# 	log9=`$bin -i $path/$dataset -m 2 -R 32 -w 4 -t 8 -l 2 | perl -p -e 's/\n//'`
# 	echo "$dataset,$log1,$log2,$log3,$log4,$log5,$log6,$log7,$log8,$log9" >> $out
# done

# # # # mode 2
# echo "mode 3" >> $out
# for dataset in uber_sorted3012.tns nips_sorted3012.tns chicago-crime-comm_sorted3012.tns flickr-4d_sorted3012.tns; do
# 	log1=`$bin -i $path/$dataset -m 3 -R 32 -w 4 -t 9 -f 16 | perl -p -e 's/\n//'`
# 	log2=`$bin -i $path/$dataset -m 3 -R 32 -w 4 -t 9 -f 32 | perl -p -e 's/\n//'`
# 	log3=`$bin -i $path/$dataset -m 3 -R 32 -w 4 -t 9 -f 64 | perl -p -e 's/\n//'`
# 	log4=`$bin -i $path/$dataset -m 3 -R 32 -w 4 -t 9 -f 128 | perl -p -e 's/\n//'`
# 	log5=`$bin -i $path/$dataset -m 3 -R 32 -w 4 -t 9 | perl -p -e 's/\n//'`
# 	log6=`$bin -i $path/$dataset -m 3 -R 32 -w 4 -t 8 | perl -p -e 's/\n//'`
# 	log7=`$bin -i $path/$dataset -m 3 -R 32 -w 4 -t 8 -f 128 | perl -p -e 's/\n//'`
# 	log8=`$bin -i $path/$dataset -m 3 -R 32 -w 4 -t 8 -f 128 -l 2| perl -p -e 's/\n//'`
# 	log9=`$bin -i $path/$dataset -m 3 -R 32 -w 4 -t 8 -l 2 | perl -p -e 's/\n//'`
# 	echo "$dataset,$log1,$log2,$log3,$log4,$log5,$log6,$log7,$log8,$log9" >> $out
# # # rm tmp && make && ./mttkrp -i ../dataset/3d_3_8_sorted012.tns -m 0 -R 4 -w 2 -t 8 -o tmp && cat tmp

