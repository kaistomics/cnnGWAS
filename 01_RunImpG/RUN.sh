#!/bin/bash

#############################################
DIR=`pwd`
DIR_RawDatas=$DIR/../00_RawDatas/
RAWFILE_list="$@"                  #AUTsummary_hg19.txt

DIR_tmp=$DIR/Data/datas
DIR_out=$DIR/Data/

IMPG_Utils=$DIR/ImpG/ImpG-Utils/
IMPG_Bins=$DIR/ImpG/ImpG-Bins/
##############################################
if [ ! -d $DIR_tmp ]; then mkdir -p $DIR_tmp; fi

###################
# 00 Install ImpG
if [ ! -d ImpG/ImpG-Bins/ ]
  then git clone https://github.com/huwenboshi/ImpG
  cd ImpG/ImpG-Bins
  make
  cd -
fi

####################
# 01 ImpG-settings
bash 00_ImpG_settings_EUR.sh $DIR_tmp $IMPG_Utils

########################
# 02 RUN Imputation 
for RAWFILE in ${RAWFILE_list}
 do
 DIS=${RAWFILE%.txt}
 F_zscore=${DIR_RawDatas}/${RAWFILE}
 DIR_out_disease=${DIR_out}/${DIS}
 if [ ! -d $DIR_out_disease ]; then mkdir -p $DIR_out_disease; fi

 bash 01_ImpG_run.sh $F_zscore $DIR_tmp $DIR_out_disease ${IMPG_Utils} ${IMPG_Bins} 
 done
