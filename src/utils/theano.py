from theano import shared, config
import theano
import numpy as np

def shared_floatx(shape):
    return theano.shared(np.zeros(shape, dtype=config.floatX))

def shared_rnd(shape, rng):
    vl = np.asarray(
            np.random.uniform(
                low = rng[0],
                high = rng[1],
                size = shape
            ),
            dtype = config.floatX
        )
    return shared(value=vl, borrow=True)

