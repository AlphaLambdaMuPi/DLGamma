from preprocessing import *
from profile import BaseProfile
from settings import *
from os.path import join as pjoin
import os
import numpy as np
import theano.tensor as T, theano
from theano import config
from dnn.layer import Recurrent
from dnn.cost import cosine_regression_cost, cosine_distance, log_regression_cost
from scipy.spatial.distance import cosine as cosine_d
from progressbar import ProgressBar
import pickle


class Profile(BaseProfile):
    def start(self):

        DIMS = [2, 3, 4]
        x = T.tensor3('x')
        y = T.ivector('y')

        layers = [Recurrent(T.tanh), Recurrent(T.nnet.softmax)]
        layers[0].initialize(input_dim = DIMS[0], output_dim = DIMS[1], hidden_dim = 2)
        layers[1].initialize(input_dim = DIMS[1], output_dim = DIMS[2], hidden_dim = 2)

        yhat = layers[1].apply(layers[0].apply(x.dimshuffle(1, 0, 2)))[-1]
        cost = log_regression_cost(y, yhat)
        #cost = T.mean(T.nnet.categorical_crossentropy(yhat, y))
        D = [[[1,2],[3,4],[5,6]], [[9,10],[7,8],[5,6]]]

        test_func = theano.function([x, y], [yhat, cost])
        for lr in layers: lr.init_state(2)
        print(test_func(
            np.array( D , dtype=config.floatX) ,
            np.array( [0, 1], dtype=np.int32)))
        for lr in layers: lr.init_state(1)
        print(test_func(
            np.array( D[0:1] , dtype=config.floatX) ,
            np.array([0], dtype=np.int32)))
        print(test_func(
            np.array( D[1:2] , dtype=config.floatX) ,
            np.array([1], dtype=np.int32)))

        eta = 0.1
        params = layers[0].params + layers[1].params
        updates = [ (p, p - eta*g) for p, g in zip(params, T.grad(cost, params)) ]
        train_func = theano.function([x, y], [yhat, cost], updates = updates)

        while True:
            for lr in layers: lr.init_state(2)
            print(train_func(
                np.array( D , dtype=config.floatX) ,
                np.array( [0, 1], dtype=np.int32)))
            print(test_func(
                np.array( D , dtype=config.floatX) ,
                np.array( [0, 1], dtype=np.int32)))
            for lr in layers: lr.init_state(1)
            print(test_func(
                np.array( D[0:1] , dtype=config.floatX) ,
                np.array([0], dtype=np.int32)))
            print(test_func(
                np.array( D[1:2] , dtype=config.floatX) ,
                np.array([1], dtype=np.int32)))
            input()

        


        


