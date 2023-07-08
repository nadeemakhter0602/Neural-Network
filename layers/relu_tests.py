import unittest
import numpy as np
from relu import ReLU


class LeakyReLUTests(unittest.TestCase):
    def setUp(self):
        self.testing = np.testing

    def test_relu(self):
        layer = ReLU()
        result = layer.forward(np.array([-3.0, -1.0, 0.0, 2.0]))
        expected = np.array([0.0, 0.0, 0.0, 2.0])
        self.testing.assert_array_almost_equal(expected, result)

    def test_relu_with_max_value(self):
        layer = ReLU(max_value=1.0)
        result = layer.forward(np.array([-3.0, -1.0, 0.0, 2.0]))
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        self.testing.assert_array_almost_equal(expected, result)

    def test_relu_with_negative_slope(self):
        layer = ReLU(negative_slope=1.0)
        result = layer.forward(np.array([-3.0, -1.0, 0.0, 2.0]))
        expected = np.array([-3.0, -1.0, 0.0, 2.0])
        self.testing.assert_array_almost_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
