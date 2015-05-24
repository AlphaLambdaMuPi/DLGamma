from settings import *
import pickle
from os.path import join as pjoin
from collections import Counter

def generate_all(max_occ=5, max_len=20):
    with open(DATA['word_list_test'], 'rb') as f:
        st = set(pickle.load(f))

    name = 'train0'
    c = Counter()
    with open(DATA['train_sentences']) as f, \
         open(pjoin(PATH['train_data'], '{}.txt'.format(name)), 'w') as fw:
        mx = 0
        for ln in f:
            ls = ln.strip('\n').split()
            for i, x in enumerate(ls):
                if i > max_len: break
                if x in st and c[x] < max_occ:
                    s = ' '.join(ls[:i])
                    s += ' [{}]\n'.format(x)
                    mx = max(i, mx)
                    fw.write(s)
                    c[x] += 1
    with open(pjoin(PATH['train_data'], '{}.desc'.format(name)), 'w') as fw:
        fw.write('Maxlen = {}'.format(mx))


