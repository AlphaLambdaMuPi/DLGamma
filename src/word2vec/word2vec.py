from settings import *
import numpy as np
import pickle

with open(DATA['word_list'], 'rb') as wlf:
    wl = pickle.load(wlf)

M = {}
for i, x in enumerate(wl):
    M[x] = i

V = np.load(DATA['word2vec.npy'])

def word2vec(w):
    return V[M[w]]

