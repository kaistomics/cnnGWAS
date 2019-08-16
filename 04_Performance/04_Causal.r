#!/usr/bin/R

options(stringsAsFactors=F)
args = commandArgs(trailingOnly = T)
DIR_Prep=args[1]
DIR_CNN =args[2]
fname   =args[3]
DIS     =args[4:length(args)]
library(ggplot2)
library(reshape2)
library(grid)
library(gridExtra)
library(gplots)
library(plyr)

for (DISEASE in DIS){

## READ FEATURE NAME ##
 featNames = unlist(read.csv(pipe(paste("sed -n 1p ",DIR_Prep,"/pred_",DISEASE,"_f0.05.tsv",sep="")),header=F,sep=''))[-c(1:2)]

## READ CAUSAL PROBABILITY ##
 f = read.csv(pipe(paste('python predict_Causal.py',DIR_Prep, DIR_CNN, DISEASE,'|sort|uniq')),header=F,sep='')
 colnames(f)=c('CHR','pos','tagSNP','features','prob')
 f=f[,-4]

## WRITE TABLE ##
 write.table(f, paste(fname, DISEASE, ".txt", sep=''), col.names=F, row.names=F, quote=F, sep="\t")

}

