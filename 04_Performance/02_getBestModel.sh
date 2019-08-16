#!/bin/bash

DIR=$1 # AUT_out
cat <(for file in $(ls $DIR/*log); do printf $file"\t"; grep 'Best.*AUC' $file; done) |grep _AE_True_ |awk '{if(NF==22){print }}'|awk '{print $1"\t"$(NF)}' | sort -nrk2 |cut -f1 | head -1 
cat <(for file in $(ls $DIR/*log); do printf $file"\t"; grep 'Best.*AUC' $file; done) |grep _AE_False_|awk '{if(NF==22){print }}'|awk '{print $1"\t"$(NF)}' | sort -nrk2 |cut -f1 | head -1  
