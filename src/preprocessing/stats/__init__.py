from settings import *
from collections import Counter
from progressbar import ProgressBar
import numpy as np
import pickle
import re

def make():
    fn = DATA['train_sentences']
    with open(fn) as f, open(DATA['stats'], 'w') as fw:
        cnt_mp = Counter()
        longest = 0
        cntw = 0
        cntl = 0
        for ln in f:
            def fun(x):
                nonlocal cnt_mp, longest, cntw
                cnt_mp[x] += 1
                longest = max(longest, len(x))
                cntw += 1
            for x in ln.strip('\n').split(): fun(x)
            cntl += 1

        fw.write(
            '''Total lines = {}
               Total words = {}
               Total diffents words = {}
               Maximum length words = {}
            '''.format(cntl, cntw, len(cnt_mp), longest))
        fw.write('Most 100 freq. words = \n')
        for w, f in cnt_mp.most_common(100):
            fw.write('{} : {}\n'.format(w, f))

    vecmap = {}
    wd_list = []
    with open(DATA['word2vec']) as w2vf, ProgressBar(maxval=3000200) as pg:
        cnt = 0
        z = 0
        for l in w2vf:
            arr = l.strip('\n').split()
            if arr[0] in cnt_mp and arr[0] not in vecmap:
                _shape = len(arr) - 1
                z += 1
                v = np.hstack((
                    np.array(list(map(float, arr[1:]))),
                    np.array([0.])
                ))
                vecmap[arr[0]] = v
            cnt += 1
            pg.update(cnt)
        print('{} words in vec'.format(z))

    _ls = []
    for w in cnt_mp:
        v = vecmap.get(w, False)
        if v is not False:
            _ls.append(v)
        else:
            _ls.append(np.array([0.] * _shape + [1.]))
        wd_list.append(w)

    with open(DATA['word_list'], 'wb') as wlf:
        pickle.dump(wd_list, wlf)
    np.save(DATA['word2vec.npy'], np.asarray(_ls)) 

def make_test():
    with open(DATA['questions']) as f:
        regex = re.compile('\[(.+)\]')
        st = set()
        while True:
            l = f.readline()
            if not l: break
            s = regex.search(l).group(1)
            st.add(s)
            for i in range(1, 5):
                l = f.readline()
                s = regex.search(l).group(1)
                st.add(s)
        ls = list(st)
        with open(DATA['word_list_test'], 'wb') as wlf:
            pickle.dump(ls, wlf)

