#!/bin/bash
###################################################################
DIR=`pwd`
DIR_raw=$DIR/../00_RawDatas/
DIR_imp=$DIR/../01_RunImpG/Data/
DIR_block=$DIR/SigBlock

RAWFILE_list="$@"  #AUTsummary_hg19.txt
#DIS_LIST=("AUT")
#FEAT_LIST=("DHS349" "HISTONE" "Pathway")
DB_DIR="../DB/"
N=4 #The number of Features
FEAT[1]="DHS349";               DIR_DB[1]=$DB_DIR/DHS_349
FEAT[2]="HISTONE";              DIR_DB[2]=$DB_DIR/HISTONE
FEAT[3]="Pathway";              DIR_DB[3]=$DB_DIR/Pathway
FEAT[4]="FIMO";                 DIR_DB[4]=$DB_DIR/FIMO
DB_featNames=$DB_DIR/featNames
EP_MAP_FILE=$DB_DIR/DHSCorr/Dhs_Fantom_Prom.refGene.bed
SigPVal=5e-04
CUTOFF=0.05
##################################################################
# Make Features names
DIR_tmp='./Data'
if [ ! -d $DIR_tmp ];      then mkdir -p $DIR_tmp;      fi
if [ ! -d $DB_featNames ]; then mkdir -p $DB_featNames; fi

for ((i=1; i<=$N; i++)); do
 F_FEAT=$DB_featNames/featNames.${FEAT[$i]}.txt
 rm -f $F_FEAT
 ls ${DIR_DB[$i]}/*bed |while read FILE; do
  echo $(basename ${FILE%.bed}) >> $F_FEAT
 done
done
##################################################################
# RUN

for RAWFILE in ${RAWFILE_list}
do
 DIS=${RAWFILE%.txt}
 for ((i=1; i<=$N; i++)); do
  OUT_TB[$i]=$DIR_tmp/result.${DIS}_${FEAT[$i]}.tsv
 done
 #####################
 # 00 Get Sig Blocks
 F_raw=$DIR_raw/${RAWFILE}
 if [ ! -d $DIR_block     ]; then mkdir -p $DIR_block;    fi
 Rscript 00_FindSigBlocks.r $DIS $F_raw $DIR_imp $DIR_block $SigPVal

 #####################
 # 01 AssociationBlock -> SNP_BED files
 CASE_INPUT=$DIR_block/${DIS}.input.txt
 bash 01_make_input.sh $DIS $CASE_INPUT $EP_MAP_FILE $DB_DIR $DIR_tmp

  #####################
  # 02 SNP_BED -> Intersect with DB_features 
 SNP_BED=$DIR_tmp/${DIS}.sorted.bed
 for ((i=1; i<=$N; i++)); do
  bash   02_make_train_tb.forSNP.sh $SNP_BED ${DIR_DB[$i]} ${OUT_TB[$i]}
  #####################
  # 03 Filtering
  F_feat=$DB_featNames/featNames.${FEAT[$i]}.txt
  python 03_make_train_filt.py ${OUT_TB[$i]} $F_feat $CUTOFF
 done

 ######################
 # 04 Merge Feat Files
 echo "*** #make pred files ***"
 paste       ${OUT_TB[1]%.tsv}_f${CUTOFF}.tsv  \
       <(cat ${OUT_TB[2]%.tsv}_f${CUTOFF}.tsv |cut -f1,2 --complement) \
       <(cat ${OUT_TB[3]%.tsv}_f${CUTOFF}.tsv |cut -f1,2 --complement) \
       <(cat ${OUT_TB[4]%.tsv}_f${CUTOFF}.tsv |cut -f1,2 --complement) > $DIR_tmp/pred_${DIS}_f${CUTOFF}.tsv
done


for RAWFILE in ${RAWFILE_list[*]}
do
 DIS=${RAWFILE%.txt}
 # 05 Make Pkl file
 echo "*** PROCESS #make pkl files ***"
 F_PICKLE=$DIR_tmp/${DIS}_AE_30SNP.pkl.gz
 python 04_make_trainAE_pkl.py $DIS $CUTOFF $F_PICKLE $DIR_tmp ${FEAT[1]} ${FEAT[2]} ${FEAT[3]} ${FEAT[4]}
done

