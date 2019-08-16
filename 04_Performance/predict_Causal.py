#!/usr/bin/python
from CausalCNN import *
import math
import subprocess
import sys
import numpy as np
import pandas as pd

DIR_Prep=sys.argv[1]
DIR_CNN =sys.argv[2]
DISEASE =sys.argv[3] # AUT

def sigmoid(x):   return 1 / (1 + np.exp(-x))
def relu(x):     return np.maximum(x, 0.)

## LOAD PARAMETERS FROM BEST MODEL ##
best_model= open(DIR_CNN+'/'+DISEASE+'_out/BestModels.txt').readlines()[0].split('.log')[0]+'.params.pkl.gz'
w1,b1,w2,b2 = cPickle.load(gzip.open(best_model));
new_w1 = list()
for i in range(w1.shape[0]):
    new_w1.append( w1[i].flatten()[::-1] )
new_w1 = np.array(new_w1)

## MAKE GROUP DICT ##
group_dic=dict();
for line in open(DIR_Prep+'/'+DISEASE+'.bed'):
    chrom, start, end, group =line.strip().split()
    group_dic[(chrom,end)]=group

## MAKE FEATURE DICT ## 
for line in open(DIR_Prep+'/pred_'+DISEASE+'_f0.05.tsv'):
    info=line.strip().split()
    if info[0]=='SNP':
        continue
    chrom, pos = info[0].split(':')
    features=map(int, info[2:])
    print chrom, pos, group_dic[(chrom,pos)], '_'.join(map(str,features)), sigmoid( np.dot( relu( np.dot( features, new_w1.T ) +b1 ), w2.flatten()[::-1].T ) +b2 )[0]
