from preprocessing import *

from profile import BaseProfile
from settings import *
from os.path import join as pjoin
import os


class Profile(BaseProfile):
    def start(self):
        from generate import generate_all
        generate_all(name = 'train0', max_occ = 30)


