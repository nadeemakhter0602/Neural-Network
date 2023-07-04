import unittest
import numpy as np
from leakyrelu import LeakyReLU


class LeakyReLUTests(unittest.TestCase):
    def setUp(self):
        self.testing = np.testing

    def test_leakyrelu(self):
        layer = LeakyReLU()
        result = layer.forward(np.array([-3.0, -1.0, 0.0, 2.0]))
        expected = np.array([-0.9, -0.3, 0.0, 2.0])
        self.testing.assert_array_almost_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
