import numpy as np
from os.path import join as pjoin

DATA_PATH = '../data'
VEC_PATH = pjoin(DATA_PATH, 'vec.dat')

lst = list(open(VEC_PATH))[1:]
mp = {}
for sen in lst:
    sen = sen.strip('\n').split()
    x = sen[0]
    y = [float(z) for z in sen[1:]]
    mp[x] = y

def word2vec(x):
    if x not in mp: return [0.0]*200
    return mp[x]
