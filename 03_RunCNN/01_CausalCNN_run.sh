#!/bin/bash
INPUT=$1
MAX_J=40

## START
START=`date`
while read line
do
  OMP_NUM_THREADS=1 THEANO_FLAGS=device=cpu python 02_CausalCNN.py --Input $INPUT $line &
#  OMP_NUM_THREADS=1 THEANO_FLAGS=device=cpu python 02_CausalCNN_githubVer.py --Input $INPUT $line &
  sleep 5
    while (true)
    do
      NUM_J=`jobs -l|wc -l`
      if [ $NUM_J -lt $MAX_J ]
      then
         break
      fi
      sleep 2
    done
done < 00_HyperParameterSet

WORK_PID=`jobs -l |awk '{print $2}'`
wait $WORK_PID

END=`date`
echo "Start: $START"
echo "END  : $END"
