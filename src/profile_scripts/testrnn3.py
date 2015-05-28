from preprocessing import *
from profile import BaseProfile
from settings import *
from os.path import join as pjoin
import os
import numpy as np
import theano.tensor as T, theano
from theano import config
from dnn.layer import Recurrent, SpecialRecurrent
from dnn.cost import cosine_regression_cost, cosine_distance, log_regression_cost
from scipy.spatial.distance import cosine as cosine_d
from progressbar import ProgressBar
import pickle
import re


class Profile(BaseProfile):
    def start(self):
        from word2vec.word2vec import word2vec as w2v
        from word2vec.testid import w2id, words_count, id2w

        dt_name = 'trainbi'
        self.dt_name = dt_name

        OUTDIM = words_count()

        DESC_FILE = pjoin(PATH['train_data'], dt_name, 'desc.txt'.format(dt_name))
        TRAIN_FILE = pjoin(PATH['train_data'], dt_name, 'train.txt'.format(dt_name))
        TEST_FILE = pjoin(PATH['train_data'], dt_name, 'test.txt'.format(dt_name))

        with open(DESC_FILE) as f:
            l = f.readline().strip('\n')
            a,_,b = l.split()
            if a == 'Maxlen':
                mxlen = int(b)

        train_data = [[], []]
        train_target = []
        train_id = []
        CONCAT = 1
        WVDIM = 301

        lazy = 0
        with open(TRAIN_FILE) as f:
            while True:
                ln = f.readline()
                if not ln: break
                ls = list(map(lambda x: x.lower(), ln.split()))
                cur = []
                len_ = len(ls) - 1
                for i in range(mxlen - len_):
                    cur.append(np.zeros(WVDIM * CONCAT, config.floatX))
                for w in ls[:-1]:
                    v = w2v(w)
                    v /= np.linalg.norm(v)
                    cur.append(v.astype(config.floatX))
                train_data[0].append(cur)

                tg = (ls[-1])[1:-1]
                train_target.append(w2v(tg))
                train_id.append(w2id(tg))

                ln = f.readline()
                ls = list(map(lambda x: x.lower(), ln.split()))
                cur = []
                len_ = len(ls)
                for i in range(mxlen - len_):
                    cur.append(np.zeros(WVDIM * CONCAT, config.floatX))
                for w in ls:
                    v = w2v(w)
                    v /= np.linalg.norm(v)
                    cur.append(v.astype(config.floatX))
                train_data[1].append(cur)
                lazy += 1
                if lazy >= 3000:
                    break

        test_data = [[], []]
        test_options = []
        word2g = {}
        groups = []

        with open(TEST_FILE) as f:
            while True:
                ln = f.readline()
                if not ln: break
                ls = ln.split()
                ls = list(map(lambda x: x.lower(), ln.split()))
                cur = []
                len_ = len(ls) - 1
                for i in range(mxlen - len_):
                    cur.append(np.zeros(WVDIM, config.floatX))
                for w in ls[:-1]:
                    v = w2v(w)
                    v /= np.linalg.norm(v)
                    cur.append(v.astype(config.floatX))
                test_data[0].append(cur)
                opt = ls[-1][1:-1].split('@')
                curt = []
                gp = []
                for x in opt:
                    gp.append(w2id(x))
                    curt.append(w2v(x))
                    word2g[w2id(x)] = len(groups)
                groups.append(gp)
                test_options.append(curt)

                ln = f.readline()
                ls = ln.split()
                ls = list(map(lambda x: x.lower(), ln.split()))
                cur = []
                len_ = len(ls)
                for i in range(mxlen - len_):
                    cur.append(np.zeros(WVDIM, config.floatX))
                for w in ls:
                    v = w2v(w)
                    v /= np.linalg.norm(v)
                    cur.append(v.astype(config.floatX))
                test_data[1].append(cur)

        train_data = np.asarray(train_data, dtype=config.floatX)
        train_target = np.asarray(train_target, dtype=config.floatX)

        train_n = train_data.shape[1]
        perm = np.random.permutation(train_n)
        test_data[0] = np.array(test_data[0])
        test_data[1] = np.array(test_data[1])
        train_data[0] = train_data[0][perm]
        train_data[1] = train_data[1][perm]
        train_target = train_target[perm]

        test_data = np.asarray(test_data, dtype=config.floatX)
        test_options = np.asarray(test_options, dtype=config.floatX)
        test_n = test_data.shape[1]
        use_n = train_n
        use_n = 1000
        val_n = 1000

        batch_size = 8
        
        
        HDDIM = WVDIM
        #DIMS = [WVDIM*CONCAT, 1000, WVDIM]
        DIMS = [WVDIM*CONCAT, WVDIM]
        x = T.tensor3('x')
        x_rev = T.tensor3('x_rev')
        y = T.matrix('y')

        #layers = [Recurrent(T.tanh), Recurrent(T.nnet.softmax)]
        layers = [SpecialRecurrent()]
        layers[0].initialize(input_dim = DIMS[0], output_dim = DIMS[1], hidden_dim = HDDIM)
        #layers[1].initialize(input_dim = DIMS[1], output_dim = DIMS[2], hidden_dim = HDDIM)

        layers_rev = [SpecialRecurrent()]
        layers_rev[0].initialize(input_dim = DIMS[0], output_dim = DIMS[1], hidden_dim = HDDIM)

        ym = layers[0].apply(x_rev.dimshuffle(1, 0, 2))
        yhat = ym[-1]

        ym_rev = layers_rev[0].apply(x.dimshuffle(1, 0, 2))
        yhat_rev = ym_rev[-1]
        #yhat = layers[1].apply(ym)[-1]
        #cost = log_regression_cost(y, yhat)
        #y = theano.printing.Print('zzzz: ')(y)
        yf = yhat+yhat_rev
        cost = cosine_distance(y, yf)
        eta = theano.shared(np.array(0.02, dtype=config.floatX))
        eta_min = 0.002
        momentum = 0.3
        params = layers[0].params + layers_rev[0].params
        #params = layers[0].params + layers[1].params
        updates = []
        for p in params:
            pu = theano.shared(p.get_value()*0.0)
            updates.append((pu, pu * momentum + (1.-momentum) * T.grad(cost, p)))
            updates.append((p, p - eta * pu))
        updates.append((eta, T.max((eta*0.99999, eta_min))))
        updates = []
        #updates = [ (p, p - eta*g) for p, g in zip(params, T.grad(cost, params)) ]

        train_func = theano.function([x, x_rev, y], cost, updates=updates)
        test_func2 = theano.function([x, x_rev, y], [cost, yf])
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

                for lr in layers+layers_rev: lr.init_state(batch_size)
                c = train_func(train_data[0][l:r], train_data[1][l:r], train_target[l:r])
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

                for lr in layers+layers_rev: lr.init_state(r - l)
                cc, calc_y = test_func2(train_data[0][l:r], train_data[1][l:r], train_target[l:r])

                for j in range(l, r):
                    best, best_word = 100, -1
                    g = word2g[train_id[j]]
                    for k in range(5):
                        wid = groups[g][k]
                        #wid = np.random.randint(0, 10) if k != 0 else groups[g][0]
                        wv = w2v(id2w(wid))
                        dis = cosine_d(wv, calc_y[j-l])

                        print(k, wid, dis)
                        #print(dis, type(dis))
                        if dis < best:
                            best = dis
                            best_word = wid
                    if best_word == train_id[j]:
                        acc += 1

            print('Acc (In) = {0:.4f}'.format(acc/val_n))


            print('Testing ...')
            acc = 0
            loops = (test_n + batch_size - 1) // batch_size

            self.tw = []
            for i in ProgressBar()(range(loops)):
                l, r = batch_size * i, batch_size * (i+1)
                r = min(r, test_n)
                for lr in layers: lr.init_state(r-l)
                cc, calc_y = test_func2(test_data[0][l:r], test_data[1][l:r], np.ones((r-l, WVDIM), dtype=config.floatX))

                for j in range(l, r):
                    best, best_k = 100, -1
                    for k in range(5):
                        wv = test_options[j][k]
                        dis = cosine_d(wv, calc_y[j-l])
                        if dis < best:
                            best = dis
                            best_k = k
                    if best_k == 0:
                        acc += 1
                    self.tw.append(groups[j][best_k])

            print('Acc (Test) = {0:.4f}'.format(acc/test_n)) 
            
            if epoch_count > 80: break

    def end(self):
        from word2vec.testid import id2w
        OUTPUT_FILE = pjoin(PATH['train_data'], self.dt_name, 'output.txt')
        if not getattr(self, 'tw', False):
            return

        regex = re.compile('\[([^\]]*)\]')
        c = 0
        with open(DATA['test_sentences']) as f, \
                open(OUTPUT_FILE, 'w') as fw:
            fw.write('id,answer\n')
            while True:
                fg = True
                ans = -1
                for i in range(5):
                    ln = f.readline().lower()
                    if not ln:
                        fg = False
                        break
                    w = regex.search(ln).group(1)
                    if w == id2w(self.tw[c]):
                        ans = i
                if not fg: break
                zz = ['a', 'b', 'c', 'd', 'e']
                fw.write('{},{}\n'.format( c+1, zz[ans]))
                c += 1



        


        


