#!/bin/bash
############################################
F_in=$1     # F_in='./Data/AUTsummary_hg19.sorted.bed'
DIR_DB=$2   # DIR_DB='./DB/DHS_349/'
OUT_DB=$3   # OUT_DB='./Data/result.AUTsummary_hg19_DHS349.tsv'
#############################################

echo "Intersect snp and feature bed"
ls $DIR_DB/*bed |while read FILE
do
  echo -n -e "\r                                                                     "
  echo -n -e "\rProcessing $FILE........................."
  intersectBed -loj -a $F_in -b $FILE | awk '
    $5=="."{print 0} 
    $5!="."{print 1}
  ' > $(basename $FILE)_TEMP &
  . ./sleep_for_MAX_J_jobs.sh
 done
echo -e "\nWaiting..."
. ./wait_all_jobs.sh

echo "Merge snp table"
PASTE_LIST=`ls *bed_TEMP |while read FILE; do echo $FILE ;done`
python 02_paste_files.py $OUT_DB <(cat $F_in | awk '{OFS="\t";print $1":"$3, $4}') $PASTE_LIST
rm -f *bed_TEMP
