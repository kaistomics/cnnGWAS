#!/bin/bash
#####################################
Disease=$1          #Disease='AUTsummary_hg19'
CASE_INPUT=$2       #CASE_INPUT='SigBlock/AUTsummary_hg19.input.txt'
EP_MAP_FILE=$3      #EP_MAP_FILE='../DB/DHSCorr/Dhs_Fantom_Prom.refGene.bed'
DB_DIR=$4           #DB_DIR='../DB'
DB_out=$5	    #DB_out='./Data'
#####################################

BED_tmp=$DB_out/$Disease.bed
BED_sorted=$DB_out/$Disease.sorted.bed
BED_TargetGene=$DB_out/$Disease.sorted.Target.bed

cat $CASE_INPUT | awk 'BEGIN{FS=","; OFS="\t"} {for (i=3; i<=NF; i++) print $1,$i-1,$i,$2}' > $BED_tmp
cat $BED_tmp | sortBed -i  > $BED_sorted
intersectBed -a $BED_sorted -b $EP_MAP_FILE -wa -wb | cut -f1,2,3,8 | uniq > $BED_TargetGene
