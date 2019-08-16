#!/bin/bash
###########################################
RAWFILE_list="$@"
#DIS_LIST=("AUT")
DIR='../02_PreProc/Data/'
###########################################

python 00_mkHyperParameterSet.py |shuf |head -50 > 00_HyperParameterSet

for RAWFILE in ${RAWFILE_list}
do
        DIS=${RAWFILE%.txt}
 	bash ./01_CausalCNN_run.sh $DIR/${DIS}_AE_30SNP.pkl.gz 
done

