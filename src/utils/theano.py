from theano import shared
from theano.config import floatX

def shared_floatx(shape):
    return theano.shared(np.zeros(shape, dtype=floatX))
