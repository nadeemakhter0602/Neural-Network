import numpy as np


class LeakyReLU:
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, inputs):
        inputs[inputs < 0] *= self.alpha
        return inputs
