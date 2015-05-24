import os
from os.path import dirname, abspath
from os.path import join as pjoin
PATH = {}
DATA = {}

PATH['src'] = dirname(abspath(__file__))
PATH['root'] = abspath(pjoin(PATH['src'], os.pardir))
PATH['data'] = abspath(pjoin(PATH['root'], 'data'))
PATH['raw_data'] = abspath(pjoin(PATH['data'], 'Holmes_Training_Data'))
PATH['proc_data'] = abspath(pjoin(PATH['data'], 'proc_data'))
PATH['test_data'] = abspath(pjoin(PATH['data'], 'test_data'))
PATH['train_data'] = abspath(pjoin(PATH['data'], 'train_data'))
PATH['numpy'] = abspath(pjoin(PATH['data'], 'numpy'))

DATA['word2vec'] = abspath(pjoin(PATH['data'], 'google.dat'))
DATA['word2vec.npy'] = abspath(pjoin(PATH['proc_data'], 'word2vec.npy'))
DATA['word_list'] = abspath(pjoin(PATH['proc_data'], 'word_list'))
DATA['word_list_test'] = abspath(pjoin(PATH['proc_data'], 'word_list_test'))
DATA['train_sentences'] = abspath(pjoin(PATH['proc_data'], 'train_sentences.txt'))
DATA['stats'] = abspath(pjoin(PATH['proc_data'], 'stats'))
DATA['stats_test'] = abspath(pjoin(PATH['proc_data'], 'stats_test'))
DATA['questions'] = abspath(pjoin(PATH['test_data'], 'questions.txt'))

