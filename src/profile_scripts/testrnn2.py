from preprocessing import *
from profile import BaseProfile
from settings import *
from os.path import join as pjoin
import os
import numpy as np
import theano.tensor as T, theano
from theano import config
from dnn.layer import Recurrent
from dnn.cost import cosine_regression_cost, cosine_distance, log_regression_cost
from scipy.spatial.distance import cosine as cosine_d
from progressbar import ProgressBar
import pickle


class Profile(BaseProfile):
    def start(self):
        from word2vec.word2vec import word2vec as w2v
        from word2vec.testid import w2id, words_count

        dt_name = 'train0'

        OUTDIM = words_count()

        DESC_FILE = pjoin(PATH['train_data'], '{}.desc'.format(dt_name))
        TRAIN_FILE = pjoin(PATH['train_data'], '{}.txt'.format(dt_name))
        TEST_FILE = pjoin(PATH['test_data'], '{}.txt'.format(dt_name))

        with open(DESC_FILE) as f:
            l = f.readline().strip('\n')
            a,_,b = l.split()
            if a == 'Maxlen':
                mxlen = int(b)

        train_data = []
        train_target = []
        CONCAT = 3
        WVDIM = 301

        lazy = 0
        with open(TRAIN_FILE) as f:
            for ln in f:
                ls = list(map(lambda x: x.lower(), ln.split()))
                cur = []
                len_ = len(ls) - 1
                for i in range(mxlen - len_):
                    cur.append(np.zeros(WVDIM * CONCAT, config.floatX))
                for i in range(len(ls[:-1])):
                    vs = np.array([])
                    for j in range(i-CONCAT+1, i+1):
                        if j >= 0:
                            v = w2v(ls[j])
                            v /= np.linalg.norm(v)
                        else:
                            v = np.zeros(WVDIM)
                        vs = np.concatenate((vs, v))
                    cur.append(vs.astype(config.floatX))
                train_data.append(cur)
                tg = (ls[-1])[1:-1]
                #train_target.append(w2v(tg))
                train_target.append(w2id(tg))
                #lazy += 1
                if lazy > 2000: break

        test_data = []
        test_options = []
        word2g = {}
        groups = []

        with open(TEST_FILE) as f:
            for ln in f:
                ls = ln.split()
                ls = list(map(lambda x: x.lower(), ln.split()))
                cur = []
                len_ = len(ls) - 1
                for i in range(mxlen - len_):
                    cur.append(np.zeros(WVDIM*CONCAT, config.floatX))
                for i in range(len(ls[:-1])):
                    vs = np.array([])
                    for j in range(i-CONCAT+1, i+1):
                        if j >= 0:
                            v = w2v(ls[j])
                            v /= np.linalg.norm(v)
                        else:
                            v = np.zeros(WVDIM)
                        vs = np.concatenate((vs, v))
                    cur.append(vs.astype(config.floatX))
                test_data.append(cur)
                opt = ls[-1][1:-1].split('@')
                curt = []
                for x in opt:
                    curt.append(w2id(x))
                    word2g[w2id(x)] = len(groups)
                groups.append(curt)
                test_options.append(curt)

        train_data = np.asarray(train_data, dtype=config.floatX)
        train_target = np.asarray(train_target, dtype=np.int32)

        train_n = train_data.shape[0]
        perm = np.random.permutation(train_n)
        train_data = train_data[perm]
        train_target = train_target[perm]

        test_data = np.asarray(test_data, dtype=config.floatX)
        test_options = np.asarray(test_options, dtype=np.int32)
        test_n = test_data.shape[0]
        use_n = train_n
        #use_n = 1000
        val_n = 1000

        batch_size = 16
        
        
        HDDIM = 500
        DIMS = [WVDIM*CONCAT, 1000, OUTDIM]
        x = T.tensor3('x')
        y = T.ivector('y')

        layers = [Recurrent(T.tanh), Recurrent(T.nnet.softmax)]
        layers[0].initialize(input_dim = DIMS[0], output_dim = DIMS[1], hidden_dim = HDDIM)
        layers[1].initialize(input_dim = DIMS[1], output_dim = DIMS[2], hidden_dim = HDDIM)

        ym = layers[0].apply(x.dimshuffle(1, 0, 2))
        yhat = layers[1].apply(ym)[-1]
        cost = log_regression_cost(y, yhat)
        eta = theano.shared(np.array(0.02, dtype=config.floatX))
        eta_min = 0.003
        momentum = 0.3
        params = layers[0].params + layers[1].params
        updates = []
        for p in params:
            pu = theano.shared(p.get_value()*0.0)
            updates.append((pu, pu * momentum + (1.-momentum) * T.grad(cost, p)))
            updates.append((p, p - eta * pu))
        updates.append((eta, T.max((eta*0.99999, eta_min))))
        #updates = [ (p, p - eta*g) for p, g in zip(params, T.grad(cost, params)) ]

        train_func = theano.function([x, y], cost, updates=updates)
        train_func2 = theano.function([x, y], [cost, yhat], updates=updates)
        test_func = theano.function([x], yhat)
        test_func2 = theano.function([x, y], [cost, yhat])
        print('finish build...\n')

        loop_per_epoch = use_n // batch_size
        
        epoch_count = 0
        while True:
            print('Epoch #{}: '.format(epoch_count))
            epoch_count += 1
            print('training...')
            J = 0
            acc = 0
            for i in ProgressBar()(range(loop_per_epoch)):
                l, r = batch_size * i, batch_size * (i+1)

                for lr in layers: lr.init_state(batch_size)
                c = train_func(train_data[l:r], train_target[l:r])
                if np.isnan(c):
                    print('Nannanana')
                    return
                J += c

            J /= loop_per_epoch
            print('J = {:.4f}, acc = {:.4f}'.format(J, acc))

            print('validating...')

            acc = 0

            loops = (val_n + batch_size - 1) // batch_size
            for i in ProgressBar()(range(loops)):
                l, r = batch_size * i, batch_size * (i+1)
                r = min(r, val_n)

                for lr in layers: lr.init_state(r - l)
                cc, calc_y = test_func2(train_data[l:r], train_target[l:r])

                for j in range(l, r):
                    best, best_word = -100, -1
                    g = word2g[train_target[j]]
                    for k in range(5):
                        wid = groups[g][k]
                        prob = calc_y[j-l][wid]
                        if prob > best:
                            best = prob
                            best_word = wid
                    if best_word == train_target[j]:
                        acc += 1

            print('Acc (In) = {0:.4f}'.format(acc/val_n))


            print('Testing ...')
            acc = 0
            loops = (test_n + batch_size - 1) // batch_size
            for i in ProgressBar()(range(loops)):
                l, r = batch_size * i, batch_size * (i+1)
                r = min(r, test_n)
                for lr in layers: lr.init_state(r-l)
                calc_y = test_func(test_data[l:r])

                for j in range(l, r):
                    best, best_word = -100, -1
                    for k in range(5):
                        wid = test_options[j][k]
                        prob = calc_y[j-l][wid]
                        if prob > best:
                            best = prob
                            best_word = wid
                    if best_word == test_options[j][0]:
                        acc += 1

            print('Acc (Test) = {0:.4f}'.format(acc/test_n)) 
            
            if epoch_count > 80: break
        


        


