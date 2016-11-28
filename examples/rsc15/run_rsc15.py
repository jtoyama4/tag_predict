# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import gru4rec
import evaluation
import pickle

#P = '/home/toyama/tag_prediction/GRU4Rec/dataset/rsc15_test.txt'
PATH_TO_TRAIN = '/home/toyama/tag_prediction/GRU4Rec/dataset/train.pd'
PATH_TO_TEST = '/home/toyama/tag_prediction/GRU4Rec/dataset/test.pd'
dic = '../../tag.dic'
tree_path = '../../tree.dump'

if __name__ == '__main__':
    #data1 = pd.read_csv(P, sep='\t', dtype={'ItemId':np.int64})
    #valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})
    data = pd.read_pickle(PATH_TO_TRAIN)
    valid = pd.read_pickle(PATH_TO_TEST)
    with open(dic,"rb") as f:
        _tagdic = pickle.load(f)
    tagdic = {}
    for k,v in _tagdic.items():
        tagdic[v] = k
    with open(tree_path,"rb") as f:
        tree = pickle.load(f)
    
    n_hidden = 400
    print('Training GRU4Rec with {} hidden units'.format(n_hidden))
    
    gru = gru4rec.GRU4Rec(tree=tree, tagdic=tagdic, tag_to_idx=_tagdic, print_freq=1000, n_epochs=20, layers=[n_hidden], loss='cross-entropy', batch_size=50, dropout_p_hidden=0.2, learning_rate=0.01, momentum=0.0,final_act='linear')
    gru.fit(valid)
    
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
    print('Accuracy: {}'.format(res))
