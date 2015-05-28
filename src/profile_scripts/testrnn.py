from preprocessing import *
from profile import BaseProfile
from settings import *
from os.path import join as pjoin
import os
import numpy as np
import theano.tensor as T, theano
from theano import config
from dnn.layer import Recurrent
from dnn.cost import cosine_regression_cost, cosine_distance
from scipy.spatial.distance import cosine as cosine_d
from progressbar import ProgressBar
import pickle


class Profile(BaseProfile):
    def start(self):
        from word2vec.word2vec import word2vec as w2v
        dt_name = 'train0'

        with open(DATA['word_list_test'], 'rb') as f:
            wd_list = pickle.load(f)
        OUTDIM = len(wd_list)
        wd_map = {}
        for i, x in enumerate(wd_list):
            wd_map[x] = i

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
        WVDIM = 301
        with open(TRAIN_FILE) as f:
            for ln in f:
                ls = list(map(lambda x: x.lower(), ln.split()))
                cur = []
                len_ = len(ls) - 1
                for i in range(mxlen - len_):
                    cur.append(np.zeros(WVDIM, config.floatX))
                for w in ls[:-1]:
                    cur.append(w2v(w).astype(config.floatX))
                train_data.append(cur)
                tg = (ls[-1])[1:-1]
                #train_target.append(w2v(tg))
                train_target.append(wd_map[tg])

        test_data = []
        test_options = []

        #zz = 0
        with open(TEST_FILE) as f:
            for ln in f:
                #zz += 1
                #if zz > 100: break
                ls = ln.split()
                cur = []
                len_ = len(ls) - 1
                for i in range(mxlen - len_):
                    cur.append(np.zeros(WVDIM, config.floatX))
                for w in ls[:-1]:
                    cur.append(w2v(w).astype(config.floatX))
                test_data.append(cur)
                opt = ls[-1][1:-1].split('@')
                curt = []
                for x in opt:
                    curt.append(w2v(x))
                test_options.append(curt)

        train_data = np.asarray(train_data, dtype=config.floatX)
        train_target = np.asarray(train_target, dtype=config.floatX)
        test_data = np.asarray(test_data, dtype=config.floatX)
        test_options = np.asarray(test_options, dtype=config.floatX)
        train_n = train_data.shape[0]
        test_n = test_data.shape[0]
        use_n = 15000
        batch_size = 4
        
        
        HDDIM = 500
        DIMS = [WVDIM, 500, WVDIM]
        x = T.tensor3('x')
        y = T.matrix('y')
        h0 = T.matrix('h0')
        h1 = T.matrix('h1')
        #h0 = theano.shared(value=np.zeros((batch_size, HDDIM), dtype=config.floatX))

        lr = [RecurrentTanh(), RecurrentTanh()]
        lr[0].initialize(input_dim = DIMS[0], output_dim = DIMS[1], hidden_dim = HDDIM)
        lr[1].initialize(input_dim = DIMS[1], output_dim = DIMS[2], hidden_dim = HDDIM)

        [_h0, ys0], _ = theano.scan(
                fn=lr[0].apply,
                sequences = x.dimshuffle(1, 0, 2),
                outputs_info = [h0, None],
            )
        #ym = ys[-1]
        [_h1, ys1], _ = theano.scan(
                fn = lr[1].apply,
                sequences = ys0,
                outputs_info = [h1, None],
            )
        yhat = ys1[-1]
        #cost = cosine_regression_cost(y, yhat)
        cost = cosine_distance(y, yhat)
        #grads = T.grad(cost, lr.params)
        eta = 2e-2
        momentum = 0.3
        params = lr[0].params + lr[1].params
        updates = []
        for p in params:
            pu = theano.shared(p.get_value()*0.0)
            updates.append((pu, pu * momentum + (1.-momentum) * T.grad(cost, p)))
            updates.append((p, p - eta * pu))
            #updates = [ (p, p - eta*g) for p, g in zip(lr.params, grads) ]

        train_func = theano.function([x, h0, h1, y], cost, updates=updates)
        test_func = theano.function([x, h0, h1], yhat)
        test_func2 = theano.function([x, h0, h1, y], cost)
        #theano.printing.pydotprint(cost, outfile="z.png", var_with_name_simple=True)
        print('finish build...\n')

        loop_per_epoch = use_n // batch_size
        
        epoch_count = 0
        while True:
            print('Epoch #{}: '.format(epoch_count))
            epoch_count += 1
            print('training...')
            J = 0
            for i in ProgressBar()(range(loop_per_epoch)):
                if i < 20: continue
                l, r = batch_size * i, batch_size * (i+1)
                H0 = np.zeros((batch_size, HDDIM), dtype=config.floatX)
                H1 = np.zeros((batch_size, HDDIM), dtype=config.floatX)
                c = train_func(train_data[l:r], H0, H1, train_target[l:r])
                if np.isnan(c):
                    print('Nannanana')
                    return
                #print(c)

                J += c
            J /= loop_per_epoch
            print('J = {:.4f}'.format(J))
            print('validating...')

            acc = 0
            TN = 1000
            #TN = 10
            for i in ProgressBar()(range(TN)):
                H0 = np.zeros((1, HDDIM), dtype=config.floatX)
                H1 = np.zeros((1, HDDIM), dtype=config.floatX)
                calc_y = test_func(train_data[i:i+1], H0, H1)
                best, best_j = 100, -1
                for j in range(5):
                    k = (i+j)%train_data.shape[0]
                    dis = cosine_d(calc_y, train_target[k])
                    if dis < best:
                        best = dis
                        best_j = j
                if best_j == 0:
                    acc += 1
                    #print('Bingo..')

            print('Acc (In) = {0:.4f}'.format(acc/TN))

            #acc = 0
            #loops = test_n//batch_size
            #for i in range(loops):
                #l, r = i*batch_size, (i+1)*batch_size
                #H0 = np.zeros((batch_size, HDDIM), dtype=config.floatX)
                #calc_y = test_func(test_data[l:r], H0)
                #for ii in range(batch_size):
                    #best, best_j = 100, -1
                    #for j in range(5):
                        #dis = cosine_d(calc_y[ii], test_options[l+ii][j])
                        #if dis < best:
                            #best = dis
                            #best_j = j
                    #if best_j == 0: acc += 1
            #print('Acc (Test) = {0:.4f}'.format(acc/(loops*batch_size)))
            print('Testing ...')
            acc = 0
            for i in ProgressBar()(range(test_n)):
                H0 = np.zeros((1, HDDIM), dtype=config.floatX)
                H1 = np.zeros((1, HDDIM), dtype=config.floatX)
                calc_y = test_func(train_data[i:i+1], H0, H1)
                best, best_j = 100, -1
                for j in range(5):
                    dis = cosine_d(calc_y, test_options[i][j])
                    if dis < best:
                        best = dis
                        best_j = j
                if best_j == 0:
                    acc += 1
                    #print('Bingo..')
            print('Acc (Test) = {0:.4f}'.format(acc/test_n))
                
        


        


