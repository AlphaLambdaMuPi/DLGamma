import theano.tensor as T
import theano

def cosine_regression_cost(y, ybar):
    return T.mean(-T.log(1 + T.batched_dot(y, ybar) /
        (y.norm(2, axis=1) * ybar.norm(2, axis=1)) ))

def cosine_distance(y, ybar):
    return -T.mean(T.batched_dot(y, ybar) / (y.norm(2, axis=1) * ybar.norm(2, axis=1) + 0.0001))

def log_regression_cost(y, ybar):
    return -T.mean(T.log(ybar)[T.arange(y.shape[0]), y])


