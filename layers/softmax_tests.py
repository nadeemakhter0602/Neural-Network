import unittest
import numpy as np
from softmax import Softmax


class SoftmaxTests(unittest.TestCase):
    def setUp(self):
        self.testing = np.testing

    def test_relu(self):
        layer = Softmax()
        result = layer.forward(np.array([1., 2., 1.]))
        expected = np.array([0.21194156, 0.57611688, 0.21194156])
        self.testing.assert_array_almost_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
