#!/usr/bin/python
from CausalCNN import *
import math
import subprocess
import sys
import numpy  as np
import pandas as pd

DIR_Prep=sys.argv[1]
DIR_CNN =sys.argv[2]
DISEASE =sys.argv[3] 

def sigmoid(x):	return 1 / (1 + np.exp(-x))
def relu(x):	return np.maximum(x, 0.)
def getCnt(DIS, feat):
    return subprocess.Popen("head -1 "+DIR_Prep+"/result."+DIS+"_"+feat+"_f0.05.tsv |awk '{print NF-2}'", shell=True, stdout=subprocess.PIPE).stdout.readlines()[0].strip()

## LOAD PARAMETERS FROM BEST MODEL ##

BestModels =list()
for line in open(DIR_CNN+'/'+DISEASE+'_out/BestModels.txt'):
    BestModels.append(line.strip())

for best in BestModels:
    best_model  = best.split('.log')[0]+'.params.pkl.gz'
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
    groupLst=[];test_y=[];test_pred=[]
    for line in open(DIR_Prep+'/pred_'+DISEASE+'_f0.05.tsv'):
        info=line.strip().split()
        if info[0]=='SNP':
            n_DHS349 =getCnt(DISEASE,"DHS349")
            n_HISTONE=getCnt(DISEASE,"HISTONE")
            n_Pathway=getCnt(DISEASE,"Pathway")
            n_FIMO   =getCnt(DISEASE,"FIMO")  
            feat_group=['DHS349']*int(n_DHS349)+['HISTONE']*int(n_HISTONE)+['Pathway']*int(n_Pathway)+['FIMO']*int(n_FIMO)
            continue
        chrom, pos = info[0].split(':')
        chrN = int(chrom.split('chr')[1])
        if chrN not in [11,12,13,14]:
            continue
        features=map(int, info[2:])
        groupLst.append(group_dic[(chrom,pos)])
        test_y.append(1)
        test_pred.append( sigmoid( np.dot( relu( np.dot( features, new_w1.T ) +b1 ), w2.flatten()[::-1].T ) +b2 )[0] )
        for k in range(10):
            np.random.seed(0)
            rand_features= pd.DataFrame(features).groupby(feat_group).transform(np.random.permutation)[0].tolist()
            groupLst.append(group_dic[(chrom,pos)]+'_rand'+str(k+1))
            test_y.append(0)
            test_pred.append( sigmoid( np.dot( relu( np.dot( rand_features, new_w1.T ) +b1 ), w2.flatten()[::-1].T ) +b2 )[0] )

    df0 = pd.DataFrame({'test_y':test_y,'test_pred':test_pred})
    df1 = df0.groupby(groupLst).max()

    TEST_y   = np.array(df1['test_y'])
    TEST_pred= np.array(df1['test_pred'])

    TEST_auc          = ROCData( zip(TEST_y.tolist(), TEST_pred.tolist())).auc()
    TEST_sensitivity  = numpy.mean(TEST_pred[numpy.equal(TEST_y, 1)] > .5)
    TEST_specificity  = numpy.mean(TEST_pred[numpy.equal(TEST_y, 0)] <=.5)
    TEST_precision    = numpy.mean( numpy.equal(TEST_y[TEST_pred > .5],1) )
    TEST_F1           = 2./(1./TEST_precision + 1./TEST_sensitivity)

    print DISEASE, TEST_sensitivity, TEST_specificity, TEST_auc, TEST_F1

