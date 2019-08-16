#!/usr/bin/python
import sys
import numpy  as np
import pandas as pd
#######################################
F_in  =sys.argv[1] #F_in='./Data/result.AUTsummary_hg19_DHS349.tsv'
F_feat=sys.argv[2] #F_feat='../DB/featNames/featNames.DHS349.txt'
CUTOFF=sys.argv[3] #CUTOFF='0.05'
#######################################

F_out=F_in.split('.tsv')[0]+"_f"+CUTOFF+".tsv"
################
# Read featNames
featNames=[line.strip() for line in open(F_feat).readlines()]

################
# Counting by association blocks..
snps=[]; leadSNPs=[]; tb=[];
for line in open(F_in):
	info=line.strip().split('\t')
        snps.append(info[0])
        leadSNPs.append(info[1])
        tb.append(list(map(int,info[2:])))

tb_out = (pd.DataFrame(tb).groupby(leadSNPs).sum() >0).mean() > float(CUTOFF)

################
# Index Filtering
filted_index=[]
for i,v in enumerate(tb_out):
	if v==True:
		filted_index.append(i)

tb_filt   = np.array(tb)[:, filted_index]
print tb_filt.shape

################
# Write Header
fw=open(F_out,'w')
fw.write('SNP\ttagSNP\t')
fw.write('\t'.join([featNames[I] for I in filted_index])+'\n')

################
# Write Files
for i in range(len(snps)):
	fw.write(snps[i]+'\t'+leadSNPs[i]+'\t')
	fw.write('\t'.join(map(str,tb_filt[i]))+'\n')
