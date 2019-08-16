#!/bin/bash
###################################
DIR=$1  #DIR='./Data/datas'
IMPG=$2 #IMPG='./ImpG/ImpG-Utils'
##################################

wget -P $DIR/  http://143.248.31.34/~omics/data/cnnGWAS/imputation/EUR.panel 
wget -P $DIR/  http://143.248.31.34/~omics/data/cnnGWAS/imputation/ALL.chr.filt.bgl.tar.gz 
tar -zxvf $DIR/ALL.chr.filt.bgl.tar.gz -C $DIR


for i in `seq 1 22`
do
 echo $i
 ###################################
 # 4.2.2 GENERATE SNP MAPPING FILE #
  mkdir -p $DIR/chr${i}/maps/
  ${IMPG}/GenMaps -m $DIR/ALL.chr${i}.phase1_release_v3.20101123.filt.markers -p $DIR/chr${i}/maps/chr${i} -s $DIR/chr${i}/maps.names

 ################################################################################
 # 4.2.3 GENERATE HAPLOTYPE REFERENCE PANEL FILES FOR PARTITONS OF A CHROMOSOME #
  mkdir -p $DIR/chr${i}/haps
  ${IMPG}/GenHaps -d $DIR/chr${i}/maps/ -s $DIR/chr${i}/maps.names -a $DIR/ALL.chr${i}.phase1_release_v3.20101123.filt.bgl -p $DIR/EUR.panel -o $DIR/chr${i}/haps/
done

