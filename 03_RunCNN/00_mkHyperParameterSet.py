#!/usr/bin/python
import sys, os
import numpy as np

def mkParameter():
    learning_rate_    =[1.5, 1., 5e-01, 1e-01, 5e-02, 1e-02, 5e-03, 1e-03]
    pre_learning_rate_=[1.5, 1., 5e-01, 1e-01, 5e-02, 1e-02, 5e-03, 1e-03]
    L1_param_         =[0., 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1.]
    L2_param_         =[0., 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1.]
    mom_              =[0., .1, .2, .3,.4, .5]
    batch_size_       =[100]
    nkerns_	      =[50]
    corruption_level_ =[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  
    for learning_rate in learning_rate_:
        for pre_learning_rate in pre_learning_rate_:
            for L1_param in L1_param_:
                for L2_param in L2_param_:
                    for mom in mom_:
                        for batch_size in batch_size_:
                            for nkerns in nkerns_:
                                for corruption_level in corruption_level_:
                                    TMP_= ['--learning_rate', learning_rate,
                                           '--pre_learning_rate',pre_learning_rate,
                                           '--n_kerns',    nkerns,
                                           '--batch_size',  batch_size,
                                           '--L1_param',    L1_param,
                                           '--L2_param',    L2_param,
                                           '--mom',         mom,
                                           '--corruption_level',corruption_level]
                                    print ' '.join(map(str, TMP_))
###############just run one time
if __name__=='__main__':
    mkParameter()
