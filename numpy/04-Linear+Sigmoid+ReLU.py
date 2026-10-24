import numpy as np


def sigmoid(x):
    '''
    1 / (1 + np.exp(-x))
    '''

    return 1 / (1 + np.exp(-x))

def linear(x, w, b):
    '''
    x @ w + b
    '''

    return x @ w + b

def relu(x):
    '''
    np.maximum(0, x)
    '''
    return np.maximum(0, x)