from settings import *
import pickle
from os.path import join as pjoin

def generate_all():
    with open(DATA['word_list_test'], 'rb') as f:
        st = set(pickle.load(f))

    with open(DATA['train_sentences']) as f, \
         open(pjoin(PATH['train_data'], 'train0.txt'), 'w') as fw:
        for ln in f:
            ls = ln.strip('\n').split()
            for i, x in enumerate(ls):
                if x in st:
                    s = ' '.join(ls[:i])
                    s += ' []\n'
                    fw.write(s)


