import pickle
from settings import *

with open(DATA['word_list_test'], 'rb') as f:
    wd_list = pickle.load(f)
    wd_list = list(map(lambda x: x.lower(), wd_list))

wd_map = {}
for i, x in enumerate(wd_list):
    wd_map[x] = i

def words_count():
    return len(wd_list)

def w2id(x):
    return wd_map[x]

def id2w(i):
    return wd_list[i]

