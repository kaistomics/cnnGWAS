#!/usr/bin/python

import sys
import random
import cPickle, gzip
import numpy as np
import pandas as pd
import subprocess

#############################
Disease=sys.argv[1]     #Disease='AUTsummary_hg19'
CUTOFF=sys.argv[2]      #CUTOFF='0.05'
F_PICKLE=sys.argv[3]    #F_PICKLE='AUTsummary_hg19_RandFeatCTRL_AE_5SNP.pkl.gz'
DIR=sys.argv[4]		#DIR='./Data'
FEAT_set=sys.argv[5:]   #OUT_TBfilt='./Data/ADD.DHS349_f0.05.tsv ./Data/ADD.HISTONE_f0.05.tsv ....
############################
F_BED=DIR+'/'+Disease+'.bed'
ldDic={}; ldkeyls=[]
for line in open(F_BED):
	chrom,start,pos,tagSNP =line.strip().split('\t')
	if tagSNP in ldDic:
		ldDic[tagSNP].append(chrom+':'+pos)
	else:
		ldDic[tagSNP]=[chrom+':'+pos]
	ldkeyls.append(tagSNP)

# SNP set
ldkeyls =list(set(ldkeyls))
for key in ldkeyls:
	cnt=len(ldDic[key])
	ldDic[key]=ldDic[key]+[ldDic[key][-1] for i in range(30-cnt)]

ldDic_train={};   ldDic_valid={};   ldDic_test={}
ldkeyls_train=[]; ldkeyls_valid=[]; ldkeyls_test=[]
for key in ldkeyls:
	chrN=int(key.split(':')[0].split('chr')[1])
	if chrN in [1,2,3,4,5,6,7]:
		ldDic_train[key]=ldDic[key]
		ldkeyls_train.append(key)
	else:
		if chrN in [15,16,17,18,19,20,21,22]:
			ldDic_test[key]=ldDic[key]
			ldkeyls_test.append(key)
		elif chrN in [8,9,10]:
			ldDic_valid[key]=ldDic[key]
			ldkeyls_valid.append(key)
####################################### Auto-encoder Disease
F_pred=DIR+'/pred_'+Disease+'_f'+CUTOFF+'.tsv'
AE = list()
for line in open(F_pred):
	info = line.strip().split()
	key = info[0]
        if key=='SNP':
            continue
        chrN = int(key.split(':')[0].split('chr')[1])
        if chrN in [1,2,3,4,5,6,7]:
		feat = info[2:]
		AE.append(feat)
#######################################
def makeDataset(FEAT_set, feat_group, ldDic, ldkeyls):
        arrayls=[];
        featDic={}
        for i in range(len(FEAT_set)):
		F_result=DIR+'/result.'+Disease+'_'+FEAT_set[i]+'_f'+CUTOFF+'.tsv'
                lines=open(F_result).readlines()
                for line in lines:
                        info=line.strip().split('\t')
                        if info[0] =='SNP':
                                continue
                        featDic[info[0]]=list(map(int,info[2:]))
                setls=[]
                for key in ldkeyls: 
                        ldsetls=[]
                        for snp in ldDic[key]: 
                                ldsetls.append(featDic[snp])
                        setls+=ldsetls
                arrayls.append(np.array(setls))
        caseArray=np.concatenate(arrayls, axis=1) 
        caseDf   =pd.DataFrame(caseArray.T)
        ctrlArray=[]
        for i in range(10):
		ctrlDf = caseDf.groupby(feat_group).transform(np.random.permutation)
                ctrlArray.append(np.array(ctrlDf).T)
        data_x, data_y = [],[]  #[feature, target]
        SNPsetN=caseArray.shape[0]/30
        for i in range(SNPsetN):
                data_x.append(list(np.transpose(caseArray[i*30:(i+1)*30,:]).flatten()))
                data_y.append(1)
        for k in range(10):
                for i in range(SNPsetN):
                        data_x.append(list(np.transpose(ctrlArray[k][i*30:(i+1)*30,:]).flatten()))
                        data_y.append(0)
        return (np.array(data_x), np.array(data_y))
###
def getCnt(Disease,feat):
	F_tmp=DIR+"/result."+Disease+"_"+feat+"_f"+CUTOFF+".tsv"
	cmd="head -1 "+F_tmp+" |awk '{print NF-2}'"
	return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.readlines()[0].strip()

n_DHS349 =getCnt(Disease,"DHS349")
n_HISTONE=getCnt(Disease,"HISTONE")
n_Pathway=getCnt(Disease,"Pathway")
n_FIMO   =getCnt(Disease,"FIMO")

feat_group=['DHS349']*int(n_DHS349)+['HISTONE']*int(n_HISTONE)+['Pathway']*int(n_Pathway)+['FIMO']*int(n_FIMO)

train =makeDataset(FEAT_set, feat_group, ldDic_train, ldkeyls_train)
valid =makeDataset(FEAT_set, feat_group, ldDic_valid, ldkeyls_valid)
test  =makeDataset(FEAT_set, feat_group, ldDic_test,  ldkeyls_test)
nSNP=30
nFeat=test[0].shape[1]/nSNP
cPickle.dump([(nFeat, nSNP), np.array(AE), train, valid, test],gzip.open(F_PICKLE,'wb'), protocol=2)

print F_PICKLE+" Written.. ("+str(nFeat)+" features were used)"
print len(ldkeyls)
print np.array(AE).shape
print len(train[1])/11.
print len(valid[1])/11.
print len(test[1])/11.
