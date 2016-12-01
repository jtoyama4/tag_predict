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
PATH_TO_TRAIN = '/home/toyama/tag_prediction/GRU4Rec/dataset/train_one_tag.pd'
PATH_TO_TEST = '/home/toyama/tag_prediction/GRU4Rec/dataset/test_one_tag.pd'
dic = '../../tag.dic'
tree_path = '../../tree.dump'

if __name__ == '__main__':
    #data1 = pd.read_csv(P, sep='\t', dtype={'ItemId':np.int64})
    #valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})
    data = pd.read_pickle(PATH_TO_TRAIN)
    valid = pd.read_pickle(PATH_TO_TEST)
    itemids = data["tag"].unique()
    
    with open(dic,"rb") as f:
        _tagdic = pickle.load(f)
    tagdicc = {}
    for k,v in _tagdic.items():
        tagdicc[v] = k
    tagdic = {}
    _tagdic = {}
    for n,i in enumerate(sorted(itemids)):
        tagdic[n] = tagdicc[i]
        _tagdic[tagdicc[i]] = n
    erases = []
    for n,(k,r) in enumerate(valid.iterrows()):
        if r['tag'] not in itemids:
            erases.append(n)
    valid = valid.drop(erases)

    with open(tree_path,"rb") as f:
        tree = pickle.load(f)
    
    n_hidden = 30
    print('Training GRU4Rec with {} hidden units'.format(n_hidden))
    
    gru = gru4rec.GRU4Rec(tree=tree, tagdic=tagdic, tag_to_idx=_tagdic, print_freq=100, n_epochs=1, layers=[30], loss='cross-entropy', batch_size=50, dropout_p_hidden=0.2, learning_rate=0.1, momentum=0.0,final_act='linear')
    gru.fit(valid,max_len = len(tagdic))
    
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
    print('Accuracy: {}'.format(res))
