import numpy as np


class LeakyReLU:
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, inputs):
        inputs[inputs < 0] *= self.alpha
        return inputs


if __name__ == '__main__':
    leakyrelu = LeakyReLU()
    output = leakyrelu.forward(np.array([-3.0, -1.0, 0.0, 2.0]))
    print(output)  # [-0.9, -0.3, 0.0, 2.0]
    leakyrelu = LeakyReLU(alpha=0.1)
    output = leakyrelu.forward(np.array([-3.0, -1.0, 0.0, 2.0]))
    print(output)  # [-0.3, -0.1, 0.0, 2.0]
