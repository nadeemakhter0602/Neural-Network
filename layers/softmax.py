import numpy as np


class Softmax():
    def forward(self, inputs, axis=-1):
        exponent = np.exp(inputs - np.max(inputs, axis=axis, keepdims=True))
        return exponent / exponent.sum(axis=axis, keepdims=True)


if __name__ == '__main__':
    softmax = Softmax()
    inp = np.asarray([1., 2., 1.])
    print("Test input :", inp)
    out = softmax.forward(inp)
    print("Output :", out)  # [0.21194156 0.57611688 0.21194156]
