#!/bin/bash
##############################################
F_zscore=$1   #F_zscore='../00_RawDatas/AUT.raw.hg19.txt'
DIR_tmp=$2    #DIR_tmp='./Data/datas'
DIR_out=$3    #DIR_out='./Data/AUT'
IMPG_Utils=$4 #IMPG_Utils='./ImpG/ImpG-Utils/'
IMPG_Bins=$5  #IMPG_Bins='./ImpG/ImpG-Bins/'
##############################################

for i in `seq 1 22`
 do
 mkdir -p ${DIR_out}/chr${i}
 
  ln -s ${DIR_tmp}/chr${i}/maps       ${DIR_out}/chr${i}/maps
  ln -s ${DIR_tmp}/chr${i}/haps       ${DIR_out}/chr${i}/haps
  ln -s ${DIR_tmp}/chr${i}/maps.names ${DIR_out}/chr${i}/maps.names
 done

cat $F_zscore | grep -v snpid | awk -v DIR_out=$DIR_out '{OFS="\t";print $1,$3,$4,$5,$6 >> DIR_out"/"$2"/zsc.txt"}'

cd ${DIR_out}

for i in `seq 1 22`
 do
 #############################################
 # 4.2.4 PARTITION SNPS FROM ANNOTATION FILE # 
 mkdir -p chr${i}/typed
 ${IMPG_Utils}/GenMaps-Typed -m chr${i}/maps/ -s chr${i}/maps.names -y chr${i}/zsc.txt -d chr${i}/typed/ -o chr${i}/maps.names.2

 #####################################################################
 # 4.2.5 GENERATE BETA FILES FOR ALL THE PARTITILS OF ONE CHROMOSOME #
 mkdir -p chr${i}/betas
 ${IMPG_Utils}/ImpG-Summary-GenBeta-Chr -b ${IMPG_Bins}/ImpG-Summary-GenBeta -s chr${i}/maps.names -d chr${i}/

 ########################
 # 4.2.6 IMPUTE Z-SCORE #
 mkdir -p chr${i}/imp
 ${IMPG_Utils}/ImpG-Summary-Chr -b ${IMPG_Bins}/ImpG-Summary -d chr${i}/ -s chr${i}/maps.names

 ########################
 # 4.2.7 MERGE Z-SCORES #
 cat chr${i}/imp/*|grep -v SNP_name |sort -k2,2g | awk -v CHR="chr"${i} '{OFS="\t";print CHR,$2-1,$2,$1,$3,$4,$5,$6}' > all.chr${i}.impz.bed
 done

cd -
