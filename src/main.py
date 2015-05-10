import os
from os.path import join as pjoin
from preprocessing import raw_to_sentences
from settings import *

if not os.path.isdir(PATH['proc_data']):
    os.mkdir(PATH['proc_data'])

with open(pjoin(PATH['proc_data'], 'train_sentences.txt'), 'w') as f:
    for x in raw_to_sentences():
        f.write(x)

