from theano import tensor as T, config
from utils.theano import shared_floatx, shared_rnd
import numpy as np

class BaseLayer:
    def __init__(self):
        pass

    def apply(self):
        pass

    def initialize(self):
        pass

class Linear(BaseLayer):
    def initialize(self, input_dim, output_dim, init_w_func, init_b_func):
        self.W = shared_floatx((output_dim, input_dim))
        self.b = shared_floatx(output_dim)

        init_w_func(self.W)
        init_b_func(self.b)

    def apply(self, inp):
        return T.dot(self.W, inp) + self.b

class LinearWithActivation(Linear):
    def initialize(self, input_dim, output_dim, init_w_func, init_b_func, 
                   activate_func):
        super().initialize(input_dim, output_dim, init_w_func, init_b_func)
        self.activate = activate_func

    def apply(self, inp):
        return self.activate(super().apply(inp))

class Sigmoid(LinearWithActivation):
    def initialize(self, input_dim, output_dim, init_w_func, init_b_func):
        super().initialize(input_dim, output_dim, init_w_func, init_b_func,
                           T.nnet.sigmoid)

class RecurrentTanh(BaseLayer):
    def initialize(self, input_dim, output_dim, hidden_dim):

        def getlr(z):
            return (-np.sqrt(6./z), np.sqrt(6./z))
        self.Wxh = shared_rnd((input_dim, hidden_dim), getlr(hidden_dim+input_dim))
        self.Whh = shared_rnd((hidden_dim, hidden_dim), getlr(hidden_dim+hidden_dim))
        self.Why = shared_rnd((hidden_dim, output_dim), getlr(output_dim+hidden_dim))
        self.bh = shared_floatx(output_dim)
        self.by = shared_floatx(output_dim)

        self.params = [self.Wxh, self.Whh, self.Why, self.bh, self.by]

    def apply(self, hid, inp):
        return (
            T.tanh(T.dot(inp, self.Wxh) + T.dot(hid, self.Whh) + self.bh),
            T.tanh(T.dot(hid, self.Why) + self.by)
        )


