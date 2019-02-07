#!/bin/bash

#PBS -l nodes=1:ppn=28
#PBS -l walltime=11:59:59

bin=/users/PAS0134/osu1600/Tensor/SpMTTKRP_private/tools/densetns 

$bin dense_10_100_1000.tns 10 100 1000
$bin dense_25_250_2500.tns 25 250 2500
$bin dense_50_500_5000.tns 50 500 5000
$bin dense_500_500_500.tns 500 500 500