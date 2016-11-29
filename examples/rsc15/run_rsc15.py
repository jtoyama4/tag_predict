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

PATH_TO_TRAIN = '/home/toyama/tag_prediction/GRU4Rec/dataset/train_one_tag.pd'
PATH_TO_TEST = '/home/toyama/tag_prediction/GRU4Rec/dataset/test_one_tag.pd'
dic = '../../dataset/tags_mini.dic'
#tree_path = '../../tree.dump'


if __name__ == '__main__':
    data = pd.read_pickle(PATH_TO_TRAIN)
    valid = pd.read_pickle(PATH_TO_TEST)
    itemids = sorted(data["tag"].unique())
    
    with open(dic,"rb") as f:
        _tagdic = pickle.load(f)
    tagdicc = {}
    for k,v in _tagdic.items():
        tagdicc[v] = k
    tagdic = {}
    _tagdic = {}
    for n,i in enumerate(itemids):
        tagdic[n] = tagdicc[i]
        _tagdic[tagdicc[i]] = n
    erases = []
    for n,(k,r) in enumerate(valid.iterrows()):
        if r['tag'] not in itemids:
            erases.append(n)
    valid = valid.drop(erases)

    """with open(tree_path,"rb") as f:
        tree = pickle.load(f)
    """
    n_hidden = 400
    print('Training GRU4Rec with {} hidden units'.format(n_hidden))    
    
    gru = gru4rec.GRU4Rec(tagdic=tagdic, tag_to_idx=_tagdic, print_freq=100, n_epochs=1, layers=[n_hidden], loss='cross-entropy', batch_size=50, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0)
    gru.fit(valid)
    
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    
