import numpy as np


class ReLU():
    def __init__(self, max_value=float('inf'), negative_slope=0.0, threshold=0.0):
        self.max_value = max_value
        self.negative_slope = negative_slope
        self.threshold = threshold

    def forward(self, inputs):
        max_value = self.max_value
        negative_slope = self.negative_slope
        threshold = self.threshold
        inputs[inputs >= max_value] = max_value
        if negative_slope or threshold:
            condition = np.logical_and(threshold <= inputs, inputs < max_value)
            inputs = np.where(condition,
                              inputs,
                              negative_slope * (inputs - threshold))
        else:
            inputs[inputs < 0] = 0.0
        return inputs


if __name__ == '__main__':
    relu = ReLU()
    outputs = relu.forward(np.array([-3.0, -1.0, 0.0, 2.0]))
    print(outputs)  # [0.0, 0.0, 0.0, 2.0]
    relu = ReLU(max_value=1.0)
    outputs = relu.forward(np.array([-3.0, -1.0, 0.0, 2.0]))
    print(outputs)  # [0.0, 0.0, 0.0, 1.0]
    relu = ReLU(negative_slope=1.0)
    outputs = relu.forward(np.array([-3.0, -1.0, 0.0, 2.0]))
    print(outputs)  # [-3.0, -1.0, 0.0, 2.0]
    relu = ReLU(threshold=1.5)
    outputs = relu.forward(np.array([-3.0, -1.0, 1.0, 2.0]))
    print(outputs)  # [0.0, 0.0, 0.0, 2.0]
