import os
from os.path import dirname, abspath
from os.path import join as pjoin
PATH = {}

PATH['src'] = dirname(abspath(__file__))
PATH['root'] = abspath(pjoin(PATH['src'], os.pardir))
PATH['data'] = abspath(pjoin(PATH['root'], 'data'))
PATH['raw_data'] = abspath(pjoin(PATH['data'], 'Holmes_Training_Data'))
PATH['proc_data'] = abspath(pjoin(PATH['data'], 'proc_data'))
PATH['test_data'] = abspath(pjoin(PATH['data'], 'test_data'))
