# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:14:20 2015
@author: Balázs Hidasi
"""

import theano
from theano import tensor as T
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import pandas as pd
from collections import OrderedDict
srng = RandomStreams()
import time
class GRU4Rec:
    '''
    GRU4Rec(layers, n_epochs=10, batch_size=50, dropout_p_hidden=0.5, learning_rate=0.05, momentum=0.0, adapt='adagrad', decay=0.9, grad_cap=0, sigma=0, init_as_normal=False, reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None, train_random_order=False, lmbd=0.0, session_key='SessionId', item_key='ItemId', time_key='Time')
    Initializes the network.

    Parameters
    -----------
    layers : 1D array
        list of the number of GRU units in the layers (default: [100] --> 100 units in one layer)
    n_epochs : int
        number of training epochs (default: 10)
    batch_size : int
        size of the minibacth, also effect the number of negative samples through minibatch based sampling (default: 50)
    dropout_p_hidden : float
        probability of dropout of hidden units (default: 0.5)
    learning_rate : float
        learning rate (default: 0.05)
    momentum : float
        if not zero, Nesterov momentum will be applied during training with the given strength (default: 0.0)
    adapt : None, 'adagrad', 'rmsprop', 'adam', 'adadelta'
        sets the appropriate learning rate adaptation strategy, use None for standard SGD (default: 'adagrad')
    decay : float
        decay parameter for RMSProp, has no effect in other modes (default: 0.9)
    grad_cap : float
        clip gradients that exceede this value to this value, 0 means no clipping (default: 0.0)
    sigma : float
        "width" of initialization; either the standard deviation or the min/max of the init interval (with normal and uniform initializations respectively); 0 means adaptive normalization (sigma depends on the size of the weight matrix); (default: 0)
    init_as_normal : boolean
        False: init from uniform distribution on [-sigma,sigma]; True: init from normal distribution N(0,sigma); (default: False)
    reset_after_session : boolean
        whether the hidden state is set to zero after a session finished (default: True)
    loss : 'top1', 'bpr' or 'cross-entropy'
        selects the loss function (default: 'top1')
    hidden_act : 'tanh' or 'relu'
        selects the activation function on the hidden states (default: 'tanh')
    final_act : None, 'linear', 'relu' or 'tanh'
        selects the activation function of the final layer where appropriate, None means default (tanh if the loss is brp or top1; softmax for cross-entropy),
        cross-entropy is only affeted by 'tanh' where the softmax layers is preceeded by a tanh nonlinearity (default: None)
    train_random_order : boolean
        whether to randomize the order of sessions in each epoch (default: False)
    lmbd : float
        coefficient of the L2 regularization (default: 0.0)
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')

    '''
    def __init__(self, layers, tree=None,tagdic=None, tag_to_idx=None,  n_epochs=10, batch_size=50, dropout_p_hidden=0.5, print_freq = 1000, learning_rate=0.05, momentum=0.0, adapt='adadelta', decay=0.9, grad_cap=0, sigma=0, init_as_normal=False, reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None, train_random_order=False, lmbd=0.0, session_key='user', item_key='tag', time_key='Time'):
        self.tagdic = tagdic
        self.tree = tree
        self.tag_to_idx = tag_to_idx
        self.print_freq = print_freq
        self.layers = layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.sigma = sigma
        self.init_as_normal = init_as_normal
        self.reset_after_session = reset_after_session
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.grad_cap = grad_cap
        self.itemtoidx={}
        self.train_random_order = train_random_order
        self.lmbd = lmbd
        if adapt == 'rmsprop': self.adapt = 'rmsprop'
        elif adapt == 'adagrad': self.adapt = 'adagrad'
        elif adapt == 'adadelta': self.adapt = 'adadelta'
        elif adapt == 'adam': self.adapt = 'adam'
        else: self.adapt = False
        if loss=='cross-entropy':
            if final_act == 'tanh':
                self.final_activation=self.softmaxth
            elif final_act == 'linear':
                self.final_activation=self.linear
            else:
                self.final_activation=self.softmax
            self.loss_function=self.cross_entropy
        elif loss=='bpr':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activation = self.relu
            elif final_act == 'sigmoid':
                self.final_activation = self.sigmoid
            else:
                self.final_activation=self.tanh
            self.loss_function=self.bpr
        elif loss=='top1':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation=self.tanh
            self.loss_function=self.top1
        else:
            raise NotImplementedError
        if hidden_act=='relu': self.hidden_activation=self.relu
        elif hidden_act=='tanh': self.hidden_activation=self.tanh
        else: raise NotImplementedError
    ######################ACTIVATION FUNCTIONS#####################
    def linear(self,X):
        return X

    def tanh(self,X):
        return T.tanh(X)

    def softmax(self,X):
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

    def softmaxth(self,X):
        X = self.tanh(X)
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

    def relu(self,X):
        return T.maximum(X, 0)

    def sigmoid(self, X):
        return T.nnet.sigmoid(X)
    #################################LOSS FUNCTIONS################################
    def cross_entropy(self, yhat):
        #yhat = T.nnet.softmax(yhat)
        #return T.cast(T.mean(-T.log(yhat)[T.arange(yhat.shape[0]),t]),theano.config.floatX)
        return T.cast(T.mean(-T.log(T.diag(yhat))), theano.config.floatX)

    def bpr(self, yhat, negative):
        return T.cast(T.mean(-T.log(T.nnet.sigmoid(yhat - negative))), theano.config.floatX)

    def top1(self, yhat):
        yhatT = yhat.T
        return T.cast(T.mean(T.mean(T.nnet.sigmoid(-T.diag(yhat)+yhatT)+T.nnet.sigmoid(yhatT**2), axis=0)-T.nnet.sigmoid(T.diag(yhat)**2)/self.batch_size), theano.config.floatX)
    ###############################################################################

    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)


    def init_weights(self, shape):
        sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (shape[0] + shape[1]))
        if self.init_as_normal:
            return theano.shared(self.floatX(np.random.randn(*shape) * sigma), borrow=True)
        else:
            return theano.shared(self.floatX(np.random.rand(*shape) * sigma * 2 - sigma), borrow=True)


    def init_matrix(self, shape):
        sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (shape[0] + shape[1]))
        if self.init_as_normal:
            return self.floatX(np.random.randn(*shape) * sigma)
        else:
            return self.floatX(np.random.rand(*shape) * sigma * 2 - sigma)


    def extend_weights(self, W, n_new):
        matrix = W.get_value()
        sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (matrix.shape[0] + matrix.shape[1] + n_new))
        if self.init_as_normal:
            new_rows = self.floatX(np.random.randn(n_new, matrix.shape[1]) * sigma)
        else:
            new_rows = self.floatX(np.random.rand(n_new, matrix.shape[1]) * sigma * 2 - sigma)
        W.set_value(np.vstack([matrix, new_rows]))


    def init(self, data):
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique()+1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        np.random.seed(42)
        self.Wx, self.Wh, self.Wrz, self.Bh, self.H = [], [], [], [], []

        for i in range(len(self.layers)):
            m = []
            m.append(self.init_matrix((self.layers[i-1] if i > 0 else self.n_items, self.layers[i])))
            m.append(self.init_matrix((self.layers[i-1] if i > 0 else self.n_items, self.layers[i])))
            m.append(self.init_matrix((self.layers[i-1] if i > 0 else self.n_items, self.layers[i])))
            self.Wx.append(theano.shared(value=np.hstack(m), borrow=True))
            self.Wh.append(self.init_weights((self.layers[i], self.layers[i])))
            m2 = []
            m2.append(self.init_matrix((self.layers[i], self.layers[i])))
            m2.append(self.init_matrix((self.layers[i], self.layers[i])))
            self.Wrz.append(theano.shared(value=np.hstack(m2), borrow=True))
            self.Bh.append(theano.shared(value=np.zeros((self.layers[i] * 3,), dtype=theano.config.floatX), borrow=True))
            self.H.append(theano.shared(value=np.zeros((self.batch_size,self.layers[i]), dtype=theano.config.floatX), borrow=True))
        self.Wy = self.init_weights((self.n_items, self.layers[-1]))
        self.By = theano.shared(value=np.zeros((self.n_items,1), dtype=theano.config.floatX), borrow=True)
        return offset_sessions


    def dropout(self, X, drop_p):
        if drop_p > 0:
            retain_prob = 1 - drop_p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX) / retain_prob
        return X


    def adam(self, param, grad, updates, sample_idx = None, epsilon = 1e-6):
        v1 = np.float32(self.decay)
        v2 = np.float32(1.0 - self.decay)
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        meang = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        countt = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = v1 * acc + v2 * grad ** 2
            meang_new = v1 * meang + v2 * grad
            countt_new = countt + 1
            updates[acc] = acc_new
            updates[meang] = meang_new
            updates[countt] = countt_new
        else:
            acc_s = acc[sample_idx]
            meang_s = meang[sample_idx]
            countt_s = countt[sample_idx]
            acc_new = v1 * acc_s + v2 * grad ** 2
            meang_new = v1 * meang_s + v2 * grad
            countt_new = countt_s + 1.0
            updates[acc] = T.set_subtensor(acc_s, acc_new)
            updates[meang] = T.set_subtensor(meang_s, meang_new)
            updates[countt] = T.set_subtensor(countt_s, countt_new)
        return (meang_new / (1 - v1**countt_new)) / (T.sqrt(acc_new / (1 - v1**countt_new)) + epsilon)


    def adagrad(self, param, grad, updates, sample_idx = None, epsilon = 1e-6):
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = acc + grad ** 2
            updates[acc] = acc_new
        else:
            acc_s = acc[sample_idx]
            acc_new = acc_s + grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling


    def adadelta(self, param, grad, updates, sample_idx = None, epsilon = 1e-6):
        v1 = np.float32(self.decay)
        v2 = np.float32(1.0 - self.decay)
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        upd = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = acc + grad ** 2
            updates[acc] = acc_new
            grad = T.sqrt(upd + epsilon) * grad
            upd_new = v1 * upd + v2 * grad ** 2
            updates[upd] = upd_new
        else:
            acc_s = acc[sample_idx]
            acc_new = acc_s + grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
            upd_s = upd[sample_idx]
            upd_new = v1 * upd_s + v2 * grad ** 2
            updates[upd] = T.set_subtensor(upd_s, upd_new)
            grad = T.sqrt(upd_s + epsilon) * grad
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling


    def rmsprop(self, param, grad, updates, sample_idx = None, epsilon = 1e-6):
        v1 = np.float32(self.decay)
        v2 = np.float32(1.0 - self.decay)
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = v1 * acc + v2 * grad ** 2
            updates[acc] = acc_new
        else:
            acc_s = acc[sample_idx]
            acc_new = v1 * acc_s + v2 * grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling


    def RMSprop(self, cost, params, full_params, sampled_params, sidxs, epsilon=1e-6):
        grads =  [T.grad(cost = cost, wrt = param) for param in params]
        sgrads = [T.grad(cost = cost, wrt = sparam) for sparam in sampled_params]
        updates = OrderedDict()
        if self.grad_cap>0:
            norm=T.cast(T.sqrt(T.sum([T.sum([T.sum(g**2) for g in g_list]) for g_list in grads]) + T.sum([T.sum(g**2) for g in sgrads])), theano.config.floatX)
            grads = [[T.switch(T.ge(norm, self.grad_cap), g*self.grad_cap/norm, g) for g in g_list] for g_list in grads]
            sgrads = [T.switch(T.ge(norm, self.grad_cap), g*self.grad_cap/norm, g) for g in sgrads]
        for p_list, g_list in zip(params, grads):
            for p, g in zip(p_list, g_list):
                if self.adapt:
                    if self.adapt == 'adagrad':
                        g = self.adagrad(p, g, updates)
                    if self.adapt == 'rmsprop':
                        g = self.rmsprop(p, g, updates)
                    if self.adapt == 'adadelta':
                        g = self.adadelta(p, g, updates)
                    if self.adapt == 'adam':
                        g = self.adam(p, g, updates)
                if self.momentum > 0:
                    velocity = theano.shared(p.get_value(borrow=False) * 0., borrow=True)
                    velocity2 = self.momentum * velocity - np.float32(self.learning_rate) * (g + self.lmbd * p)
                    updates[velocity] = velocity2
                    updates[p] = p + velocity2
                else:
                    updates[p] = p * np.float32(1.0 - self.learning_rate * self.lmbd) - np.float32(self.learning_rate) * g
        for i in range(len(sgrads)):
            g = sgrads[i]
            fullP = full_params[i]
            sample_idx = sidxs[i]
            sparam = sampled_params[i]
            if self.adapt:
                if self.adapt == 'adagrad':
                    g = self.adagrad(fullP, g, updates, sample_idx)
                if self.adapt == 'rmsprop':
                    g = self.rmsprop(fullP, g, updates, sample_idx)
                if self.adapt == 'adadelta':
                    g = self.adadelta(fullP, g, updates, sample_idx)
                if self.adapt == 'adam':
                    g = self.adam(fullP, g, updates, sample_idx)
            if self.lmbd > 0:
                delta = np.float32(self.learning_rate) * (g + self.lmbd * sparam)
            else:
                delta = np.float32(self.learning_rate) * g
            if self.momentum > 0:
                velocity = theano.shared(fullP.get_value(borrow=False) * 0., borrow=True)
                vs = velocity[sample_idx]
                velocity2 = self.momentum * vs - delta
                updates[velocity] = T.set_subtensor(vs, velocity2)
                updates[fullP] = T.inc_subtensor(sparam, velocity2)
            else:
                updates[fullP] = T.inc_subtensor(sparam, - delta)
        return updates


    def print_example(self,preds,ys,xs):
        preds = preds.T
        res = []
        for x,y,pred in zip(xs,ys,preds):
            n = 1
            tops = np.argsort(pred)[-1::-1][:n]
            print("x")
            for i in [x]:
                print(self.tagdic[i],)
            print("\nPredict")
            for i in tops:
                print(self.tagdic[i],)
            print("\nActual")
            for i in [y]:
                print(self.tagdic[i],)
            print("----------------------------------")

    def get_input(self,input_idx,y,maxlen=None):
        maxlen=self.n_items
        batch_size = len(input_idx)
        input_base = np.zeros((batch_size,maxlen),dtype="int32")
        y_base = np.zeros((batch_size,maxlen),dtype="int32")
        batch_idx = [] 
        y_idx = []
        for n,i in enumerate(input_idx):
            np.put(input_base[n],i,1)
        if y is None:
            return input_base
        for n,i in enumerate(y):
            np.put(y_base[n],i,1)
        return input_base,y_base

    def model(self, X, H, Y=None, drop_p_hidden=0.0):
        Sx = self.Wx[0][X] #TODO
        #vec = T.dot(X, self.Wx[0]) + self.Bh[0]
        vec = Sx + self.Bh[0]
        rz = T.nnet.sigmoid(vec.T[self.layers[0]:] + T.dot(H[0], self.Wrz[0]).T)
        h = self.hidden_activation(T.dot(H[0] * rz[:self.layers[0]].T, self.Wh[0]) + vec.T[:self.layers[0]].T) #CHK
        z = rz[self.layers[0]:].T
        h = (1.0-z)*H[0] + z*h
        h = self.dropout(h, drop_p_hidden)
        H_new = [h]
        y = h
        for i in range(1, len(self.layers)):
            vec = T.dot(y, self.Wx[i]) + self.Bh[i]
            rz = T.nnet.sigmoid(vec.T[self.layers[i]:] + T.dot(H[i], self.Wrz[i]).T)
            h = self.hidden_activation(T.dot(H[i] * rz[:self.layers[i]].T, self.Wh[i]) + vec.T[:self.layers[i]].T) #CHK
            z = rz[self.layers[i]:].T
            h = (1.0-z)*H[i] + z*h
            h = self.dropout(h, drop_p_hidden)
            H_new.append(h)
            y = h
        if Y is not None:
            Sy = self.Wy[Y]
            SBy = self.By[Y]
            y = self.final_activation(T.dot(y, Sy.T) + SBy.flatten())
            return H_new, y, [Sx, Sy, SBy]
            #return H_new, y
        else:
            y = self.final_activation(T.dot(y, self.Wy.T) + self.By.flatten())
            #return H_new, y, [Sx]
            return H_new, y, [Sx]

    def get_negatives(self, batch_idx, ys, data):
        counts = []
        ttt = 0
        for ii in ys:
            counts.append(len(ii))
        all_size = 0
        negative_idx = []
        num = 0
        for row,count in zip(ys,counts):
            if len(row) != count:
                print("imi fumei")
                ys[num] = row[:count]
            num += 1
            size = len(row)
            all_size += size
            if size == 0:
                continue            
            first = int(row[0])
            sample_data = self.get_sample(data)
            t = [i in sample_data for i in row]
            while any(t):
                sample_data = self.get_sample(data)
                t = [i in sample_data for i in row]
            sample_size = len(sample_data)
            if sample_size > size:
                sample_data = self.reduce_sample(sample_data,size)
            elif sample_size < size:
                sample_data = self.compensate_sample(sample_data,size)
            negative_idx.extend(sample_data)
            if size != len(sample_data):
                print("okasii")

        if len(negative_idx) != len(batch_idx):
            yyy = []
            for i in ys:
                yyy.extend(i)
            print(yy)
            print(yyy)
            #print(all_size)
            #print(ttt)
            print("get_negatives error")
            #print(len(batch_idx),len(yy),len(yyy),len(negative_idx))
            #print(y,negative_idx,batch_idx)

        return negative_idx
            
    def get_sample(self, data):
        sample_user = np.random.randint(len(data))
        sample = data.ix[sample_user,1].copy()
        return sample

    def reduce_sample(self,sample,size):
        for tag_idx in sample:
            tag = self.tagdic[tag_idx]
            if self.tree.get_node(tag) is None:
                sample.remove(tag_idx)
                if len(sample) == size:
                    return sample
        sample = sample[:size]
        return sample

    def compensate_sample(self, sample, size):
        #depth = [self.tree.depth(self.tagdic[i]) for i in sample]
        depth = []
        for i in sample:
            try:
                d = self.tree.depth(self.tagdic[i])
            except:
                d = 0
            depth.append(d)
        depth_order = np.argsort(depth)[-1::-1]
        for i in depth_order:
            tag = self.tagdic[sample[i]]
            try:
                parent = self.tag_to_idx[self.tree.parent(tag).tag]
                if parent not in sample:
                    sample.append(parent)
            except:
                pass
            if len(sample)==size:
                return sample
        n=0
        for i in depth_order:
            try:
                siblings = self.tree.siblings(self.tagdic[i])
                for sib in siblings:
                    sib_idx = self.tag_to_idx[sib.tag]
                    if sib_idx not in sample:
                        sample.append(sib_idx)
                    if len(sample) == size:
                        return sample
            except:
                pass
        while len(sample) != size:
            compen = np.random.randint(self.n_items)
            if compen in sample:
                continue
            sample.append(compen)
        return sample

    def fit(self, data, max_len=3637 ,retrain=False):
        '''
        Trains the network.

        Parameters
        --------
        data : pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        retrain : boolean
            If False, do normal train. If True, do additional train (weigths from previous trainings are kept as the initial network) (default: False)

        '''
        self.predict = None
        self.error_during_train = False
        #itemids = data[self.item_key].unique()
        itemids = data[self.item_key].unique()
        for n,i in enumerate(sorted(itemids)):
            self.itemtoidx[i] = n
        #itemids is a set of tags
        
        if not retrain:
            self.n_items = max_len
            #self.n_items = len(itemids)
            #self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
            #data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':self.itemidmap[itemids].values}), on=self.item_key, how='inner')
            offset_sessions = self.init(data)
        else:
            new_item_mask = ~np.in1d(itemids, self.itemidmap.index)
            n_new_items = new_item_mask.sum()
            if n_new_items:
                self.itemidmap = self.itemidmap.append(pd.Series(index=itemids[new_item_mask], data=np.arange(n_new_items) + len(self.itemidmap)))
                for W in [self.Wx[0], self.Wy]:
                    self.extend_weights(W, n_new_items)
                self.By.set_value(np.vstack([self.By.get_value(), np.zeros((n_new_items, 1), dtype=theano.config.floatX)]))
                self.n_items += n_new_items
                print('Added {} new items. Number of items is {}.'.format(n_new_items, self.n_items))
            data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':self.itemidmap[itemids].values}), on=self.item_key, how='inner')
            data.sort_values([self.session_key, self.time_key], inplace=True)
            offset_sessions = np.zeros(data[self.session_key].nunique()+1, dtype=np.int32)
            offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()

        #X = T.imatrix("x")
        X = T.ivector("X")
        #batch_idx = T.ivector("batch_idx")
        Y = T.ivector("Y")
        #negative_idx = T.ivector("negative_idx")
        
        H_new, Y_pred, sampled_params = self.model(X, self.H, Y, self.dropout_p_hidden)
        cost = self.loss_function(Y_pred)
        #params = [self.Wx[0], self.Wy, self.By, self.Wh[0], self.Wrz[0], self.Bh[0]]
        params = [self.Wx[1:], self.Wh, self.Wrz, self.Bh]
        full_params = [self.Wx[0], self.Wy, self.By]
        sidxs = [X, Y, Y]
        updates = self.RMSprop(cost, params, full_params, sampled_params, sidxs)
        for i in range(len(self.H)):
            updates[self.H[i]] = H_new[i]

        train_function = function(inputs=[X, Y], outputs=[cost,Y_pred], updates=updates, allow_input_downcast=True, on_unused_input='ignore')
        #train_function = function(inputs=[X, Y], outputs=[cost, Y_pred], allow_input_downcast=True, on_unused_input='ignore')
        freqs = 0

        for epoch in range(self.n_epochs):
            start_time = time.time()
            for i in range(len(self.layers)):
                self.H[i].set_value(np.zeros((self.batch_size,self.layers[i]), dtype=theano.config.floatX), borrow=True)
            c = []
            session_idx_arr = np.random.permutation(len(offset_sessions)-1) if self.train_random_order else np.arange(len(offset_sessions)-1)
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            #print(offset_sessions)            
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters]+1]
            finished = False

            while not finished:
                minlen = (end-start).min()
                out_idx = np.array([self.itemtoidx[d] for d in data.tag.values[start]],dtype="int32")
                for i in range(minlen-1):
                    in_idx = out_idx
                    out_idx = np.array([self.itemtoidx[dd] for dd in data.tag.values[start+i+1]],dtype="int32")
                    y = out_idx
                    
                    #in_idx and y must be 1-of-k shape
                    #x, _ = self.get_input(in_idx,y,maxlen=self.n_items
                    #negatives = self.get_negatives(batch_idx, y, data)
                    
                    cost,y_sample = train_function(in_idx,y)
                    
                    c.append(cost)
                    freqs += 1
                    if freqs % self.print_freq == 0:
                        self.print_example(y_sample,y,in_idx)
                    if np.isnan(cost):
                        print(str(epoch) + ': NaN error!')
                        self.error_during_train = True
                        return
                start = start+minlen-1
                mask = np.arange(len(iters))[(end-start)<=1]
                for idx in mask:
                    print(idx)
                    maxiter += 1
                    if maxiter >= len(offset_sessions)-1:
                        finished = True
                        break
                    if self.reset_after_session:
                        for i in range(len(self.H)):
                            tmp = self.H[i].get_value(borrow=True, return_internal_type=True)
                            tmp[idx,:] = 0
                            self.H[i].set_value(tmp, borrow=True)
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter]+1]
            avgc = np.mean(c)
            end_time = time.time()
            print(end_time-start_time)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            print('Epoch{}\tloss: {:.6f}'.format(epoch, avgc))

    def predict_next_batch(self, session_ids, input_item_ids, out_idxs,predict_for_item_ids=None, batch=100):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        if self.error_during_train: raise Exception
        if self.predict is None or self.predict_batch!=batch:
            X = T.imatrix('x_valid')
            #batch_idx = T.ivector('batch_idx_valid')
            y_idx = T.imatrix('y_idx_valid')
            for i in range(len(self.layers)):
                self.H[i].set_value(np.zeros((batch,self.layers[i]), dtype=theano.config.floatX), borrow=True)
            if predict_for_item_ids is not None:
                H_new, yhat, _ = self.model(X, self.H, y_idx)
            else:
                H_new, yhat = self.model(X, self.H,None)
            updatesH = OrderedDict()
            for i in range(len(self.H)):
                updatesH[self.H[i]] = H_new[i]
            if predict_for_item_ids is not None:
                self.predict = function(inputs=[X, batch_idx, y_idx], outputs=yhat, updates=updatesH, allow_input_downcast=True)
            else:
                self.predict = function(inputs=[X], outputs=yhat, updates=updatesH, allow_input_downcast=True)
            self.current_session = np.ones(batch) * -1
            self.predict_batch = batch
        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0:
            for i in range(len(self.H)):
                tmp = self.H[i].get_value(borrow=True)
                tmp[session_change] = 0
                self.H[i].set_value(tmp, borrow=True)
            self.current_session=session_ids.copy()
        in_idxs = [self.itemtoidx[d] for d in input_item_ids]
        outs = [self.itemtoidx[d] for d in out_idxs]
        #print("in_idxs", in_idxs)
        if predict_for_item_ids is not None:
            #iIdxs = self.itemidmap[predict_for_item_ids]
            x,batch_idx,y_idx = self.get_input(in_idxs, predict_for_item_ids)
            preds = np.asarray(self.predict(x, batch_idx, y_idx)).T
            return pd.DataFrame(data=preds)
        else:
            x = self.get_input(in_idxs,None,self.n_items)
            print(x.shape)
            preds = np.asarray(self.predict(x)).T
            print(preds.shape)
            print(preds)
            self.print_example(preds,outs,in_idxs)
            return pd.DataFrame(data=preds)
