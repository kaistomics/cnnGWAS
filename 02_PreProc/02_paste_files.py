#!/usr/bin/python

import sys
import numpy as np
import gzip, cPickle

if __name__ == '__main__':
    if len(sys.argv)<3:
      print " 1: out_file_name"
      print " 2...: data columns"
      exit()

    out_file   = open(sys.argv[1],'w')
    data_files = sys.argv[2:len(sys.argv)]

    #read feature table
    all_f=list()
    for i in data_files:
      f_=[line.strip() for line in open(i)]
      all_f.append(f_)

    feature_table = zip(*all_f)
    #count data
    print 'Num of features = %i' %(len(feature_table[0])-1)
    print 'Num of datasets = %i' %(len(feature_table))
    sys.stdout.flush()

    for line in feature_table:
      out_file.write('\t'.join(line)+'\n')

