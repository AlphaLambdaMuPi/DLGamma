import theano.tensor as T

def cosine_regression_cost(y, ybar):
    return T.mean(-T.log(T.batched_dot(y, ybar) /
        (y.norm(2, axis=1) * ybar.norm(2, axis=1)) ))
