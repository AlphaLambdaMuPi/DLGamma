from functools import reduce

def compose(funcs):
    return reduce(lambda f, g: lambda x: f(g(x)), funcs)

def rcompose(funcs):
    return reduce(lambda f, g: lambda x: g(f(x)), funcs)

import os
def check_path(path, create = False):
    if not os.path.isdir(path):
        if not create: return False

        os.mkdir(path)
    return True
