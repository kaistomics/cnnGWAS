#!/usr/bin/python
import glob
import sys

DIR=sys.argv[1]

def getVal(line, key):
    return line.split(key)[1].split('_')[0].split(' ')[0].split(',')[0].split('.params')[0]

fLst = glob.glob(DIR+'/*.params.pkl.gz')

print '\t'.join(['AE','LR','preLR','nKerns','batchSize','L1Param','L2Param','mom','corruption','iteration','sensitivity', 'specificity' , 'AUC','F1'])

for f in fLst:
    AE          = getVal(f,'AE_')
    learningRate= getVal(f,'learningRate_')
    pretrainLR  = getVal(f,'preLearningRate_')
    batchSize = getVal(f,'batchSize_')
    nKerns       = getVal(f,'nKerns_')
    L1Param   = getVal(f,'L1Param_')
    L2Param   = getVal(f,'L2Param_')
    mom       = getVal(f,'mom_')
    corruption= getVal(f,'corruptionLevel_')

    for line in open(f.replace('.params.pkl.gz','.log')):
        if 'Optimization complete' in line:
            test_info = line.strip()
            iteration  = getVal(test_info,'iteration ')
            sensitivity= getVal(test_info,'sensitivity ')
            specificity= getVal(test_info,'specificity ')
            AUC        = getVal(test_info,'AUC ')
            F1         = getVal(test_info,'F1 ')
            print '\t'.join([AE,learningRate, pretrainLR,nKerns, batchSize, L1Param, L2Param, mom, corruption,iteration, sensitivity, specificity , AUC, F1])
