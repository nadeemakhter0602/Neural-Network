import numpy as np


class Softmax():
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, inputs):
        axis = self.axis
        exponent = np.exp(inputs - np.max(inputs, axis=axis, keepdims=True))
        return exponent / exponent.sum(axis=axis, keepdims=True)
