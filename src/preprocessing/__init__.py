import os
from os.path import join as pjoin
from settings import *
from progressbar import ProgressBar, Bar
from utils import rcompose
import re

def preproc_train(funcs):
    files = os.listdir(PATH['raw_data'])

    with ProgressBar(maxval=len(files)) as prog:
        for cnt, filename in enumerate(files):
            with open(pjoin(PATH['raw_data'], filename), encoding='latin-1') as f:
                raw_s = f.read()
            res = rcompose(funcs)(raw_s)
            prog.update(cnt+1)
            yield res

def remove_evil_ms_linebreak(s):
    res = re.sub(r'\r', '', s)
    return res

def replace_linebreak_to_space(s):
    res = re.sub(r'\n', ' ', s)
    return res

def remove_evil_ms_linebreak(s):
    res = re.sub(r'\r', '', s)
    return res

def replace_tab_to_space(s):
    res = re.sub(r'[^\S\n]', ' ', s)
    return res

def proceed_quote(s):
    res = re.sub(r'"', r'', s)
    res = re.sub(r"'(?!(s|ve|ll))", r'', res)
    return res

def remove_reductant_space(s):
    res = s
    res = re.sub(r' +', ' ', res)
    res = re.sub(r'^ +', '', res, flags=re.M)
    res = re.sub(r' +$', '', res, flags=re.M)
    return res

def proceed_brackets(s):
    res = s
    res = re.sub(r'\(([^)]*)\)', r'\1', res)
    res = re.sub(r'\[([^)]*)\]', r'\1', res)
    res = re.sub(r'\{([^)]*)\}', r'\1', res)
    return res

def proceed_punctuation(s):
    res = s
    res = re.sub(r' ?, ?', r' , ', res)
    res = re.sub(r'-{1}', r'', res)
    res = re.sub(r' ?-{2,} ?', r' -- ', res)
    res = re.sub(r' ?([.;:?!])', r' \1\n', res)
    return res

def remove_weird_chars(s):
    res = re.sub(r'[^A-Za-z0-9\'.:;?!,]', ' ', s)
    return res
    
def to_lowercase(s):
    return s.lower()

def add_sentence_tag(s):
    res = s.split('\n')
    res = ['<s> ' + sent[:-1] + '</s>' for sent in res]
    return '\n'.join(res) 
