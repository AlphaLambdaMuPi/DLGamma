from preprocessing import *

from profile import BaseProfile
from settings import *
from os.path import join as pjoin
import os


class Profile(BaseProfile):
    desc = ''' Do all the init. '''
    def start(self):
        if not os.path.isdir(PATH['proc_data']):
            os.mkdir(PATH['proc_data'])
        funcs = [
            remove_evil_ms_linebreak,
            replace_linebreak_to_space,
            replace_tab_to_space,
            proceed_quote,
            proceed_brackets,
            remove_weird_chars,
            proceed_punctuation,
            remove_reductant_space,
            to_lowercase,
            add_sentence_tag,
        ]
        
        with open(pjoin(PATH['proc_data'], 'train_sentences.txt'), 'w') as f:
            for s in preproc_train(funcs):
                f.write(s)

        from preproc.stats import make
        make()

