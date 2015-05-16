from theano import tensor as T, config
from utils.theano import shared_floatx

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

