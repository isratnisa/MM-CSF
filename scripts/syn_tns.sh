#!/bin/bash

#PBS -l nodes=1:ppn=28
#PBS -l walltime=11:59:59

module load python/3.5

pythonFile=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/tools/generate_tensor.py

# dims0=10000
# dims1=1000000
# dims2=100000000

# rate0=`echo "scale=12 ; 300 / $dims0 * 100" | bc`
# rate1=`echo "scale=12 ; 300 / $dims1 * 100" | bc`
# rate2=`echo "scale=12 ; 300 / $dims2 * 100" | bc`
# echo $rate0, $rate1, $rate2, 

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims1=100000000
# dims2=1000000

# rate0=`echo "scale=12 ; 300 / $dims0 * 100" | bc`
# rate1=`echo "scale=12 ; 300 / $dims1 * 100" | bc`
# rate2=`echo "scale=12 ; 300 / $dims2 * 100" | bc`

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims0=1000000
# dims2=10000

# rate0=`echo "scale=12 ; 300 / $dims0 * 100" | bc`
# rate1=`echo "scale=12 ; 300 / $dims1 * 100" | bc`
# rate2=`echo "scale=12 ; 300 / $dims2 * 100" | bc`

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims1=10000
# dims2=100000000

# rate0=`echo "scale=12 ; 300 / $dims0 * 100" | bc`
# rate1=`echo "scale=12 ; 300 / $dims1 * 100" | bc`
# rate2=`echo "scale=12 ; 300 / $dims2 * 100" | bc`

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims0=100000000
# dims2=1000000

# rate0=`echo "scale=12 ; 300 / $dims0 * 100" | bc`
# rate1=`echo "scale=12 ; 300 / $dims1 * 100" | bc`
# rate2=`echo "scale=12 ; 300 / $dims2 * 100" | bc`

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims1=1000000
# dims2=10000

# rate0=`echo "scale=12 ; 300 / $dims0 * 100" | bc`
# rate1=`echo "scale=12 ; 300 / $dims1 * 100" | bc`
# rate2=`echo "scale=12 ; 300 / $dims2 * 100" | bc`

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# ### CUBE tensors

dims0=3000000
dims1=3000000
dims2=3000000

rate0=`echo "scale=12 ; 30 / $dims0 * 100" | bc`
rate1=`echo "scale=12 ; 300 / $dims1 * 100" | bc`
rate2=`echo "scale=12 ; 3000 / $dims2 * 100" | bc`

python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0"_"$rate1"_"$rate2".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

rate0=`echo "scale=12 ; 30 / $dims0 * 100" | bc`
rate1=`echo "scale=12 ; 3000 / $dims1 * 100" | bc`
rate2=`echo "scale=12 ; 300 / $dims2 * 100" | bc`

python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0"_"$rate1"_"$rate2".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2


rate0=`echo "scale=12 ; 300 / $dims0 * 100" | bc`
rate1=`echo "scale=12 ; 30 / $dims1 * 100" | bc`
rate2=`echo "scale=12 ; 3000 / $dims2 * 100" | bc`

python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0"_"$rate1"_"$rate2".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2


rate0=`echo "scale=12 ; 300 / $dims0 * 100" | bc`
rate1=`echo "scale=12 ; 3000 / $dims1 * 100" | bc`
rate2=`echo "scale=12 ; 30 / $dims2 * 100" | bc`

python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0"_"$rate1"_"$rate2".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2


rate0=`echo "scale=12 ; 3000 / $dims0 * 100" | bc`
rate1=`echo "scale=12 ; 300 / $dims1 * 100" | bc`
rate2=`echo "scale=12 ; 30 / $dims2 * 100" | bc`

python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0"_"$rate1"_"$rate2".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2


rate0=`echo "scale=12 ; 3000 / $dims0 * 100" | bc`
rate1=`echo "scale=12 ; 30 / $dims1 * 100" | bc`
rate2=`echo "scale=12 ; 300 / $dims2 * 100" | bc`

python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0"_"$rate1"_"$rate2".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2





# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims1=10000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims1=5000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims1=1000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2


# dims0=5000000
# dims1=25000000
# dims2=500000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims1=10000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims1=5000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims1=1000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# # chaning dim2

# dims0=5000000
# dims1=5000000
# dims2=25000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims2=10000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims2=5000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims2=1000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2


# dims0=5000000
# dims1=1000000
# dims2=25000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims2=10000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims2=5000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims2=1000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims0=5000000
# dims1=500000
# dims2=25000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims2=10000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims2=5000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2

# dims2=1000000

# python $pythonFile $dims0"_"$dims1"_"$dims2"_"$rate0".tns" $rate0"%"$dims0 $rate1"%"$dims1  $rate2"%"$dims2





