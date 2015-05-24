from preprocessing import *

from profile import BaseProfile
from settings import *
from os.path import join as pjoin
import os
import numpy as np
import theano.tensor as T, theano
from theano import config

from dnn.layer import RecurrentTanh
from dnn.cost import cosine_regression_cost


class Profile(BaseProfile):
    def start(self):
        from word2vec.word2vec import word2vec as w2v
        dt_name = 'train0'

        DESC_FILE = pjoin(PATH['train_data'], '{}.desc'.format(dt_name))
        TRAIN_FILE = pjoin(PATH['train_data'], '{}.txt'.format(dt_name))
        with open(DESC_FILE) as f:
            l = f.readline().strip('\n')
            a,_,b = l.split()
            if a == 'Maxlen':
                mxlen = int(b)

        train_data = []
        train_target = []
        WVDIM = 301
        with open(TRAIN_FILE) as f:
            for ln in f:
                ls = ln.split()
                cur = []
                len_ = len(ls) - 1
                for i in range(mxlen - len_):
                    cur.append(np.zeros(WVDIM, config.floatX))
                for w in ls[:-1]:
                    cur.append(w2v(w).astype(config.floatX))
                train_data.append(cur)
                tg = (ls[-1])[1:-1]
                train_target.append(w2v(tg))
        z = np.array(train_data[0])
        print(any(map(lambda x: x.shape[0] != 101 and print(x.shape[0]), z)))
        train_data = np.asarray(train_data, dtype=config.floatX)
        train_target = np.asarray(train_target, dtype=config.floatX)
        
        HDDIM = 500
        x = T.tensor3('x')
        y = T.matrix('y')
        h0 = theano.shared(value=np.zeros((1, HDDIM), dtype=config.floatX))

        lr = RecurrentTanh()
        lr.initialize(input_dim = WVDIM, output_dim = WVDIM, hidden_dim = HDDIM)

        [h, ys], _ = theano.scan(
                fn=lr.apply,
                sequences = x.dimshuffle(1, 0, 2),
                outputs_info = [h0, None],
                n_steps = x.shape[0]
            )
        yhat = ys[-1]
        cost = cosine_regression_cost(y, yhat)
        grads = T.grad(cost, lr.params)
        eta = 1e-3
        updates = { (p, p - eta*g) for p, g in zip(lr.params, grads) }

        train = theano.function([x, y], cost, updates=updates)
        print('finish build\n')
        for i in range(train_data.shape[0]):
            c = train(train_data[i].reshape((1,) + train_data[i].shape),
                      train_target[i].reshape((1,) + train_target[i].shape))
            print(c)
        


        


