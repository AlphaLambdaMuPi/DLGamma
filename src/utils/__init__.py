from functools import reduce

def compose(funcs):
    return reduce(lambda f, g: lambda x: f(g(x)), funcs)

def rcompose(funcs):
    return reduce(lambda f, g: lambda x: g(f(x)), funcs)
