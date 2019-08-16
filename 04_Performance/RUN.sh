#!/bin/bash
###########################################
RAW_filelist=("$@")
DIS_LIST=(${RAW_filelist[*]%.txt})
DIR_Prep='../02_PreProc/Data/'
DIR_CNN='../03_RunCNN/'
###########################################

##################
# Plot performance of validation set
pdf_valid=OUT_valid_performance.pdf       
Rscript 01_getAUC.r $DIR_CNN $pdf_valid ${DIS_LIST[*]}

##################
# Get Best models
for DIS in ${DIS_LIST[*]}
do
 bash 02_getBestModel.sh $DIR_CNN/${DIS}_out > $DIR_CNN/${DIS}_out/BestModels.txt
done

##################
# Plot performance of test set
pdf_test=OUT_test_performance.pdf           
Rscript 03_FinalPerformance.r $DIR_Prep $DIR_CNN $pdf_test ${DIS_LIST[*]}

##################
# Prediction score table 
txt_score=OUT_score_            
Rscript 04_Causal.r $DIR_Prep $DIR_CNN $txt_score ${DIS_LIST[*]}


