import unittest
from sklearn_tools.misc import JSONToObjectTransformer
import numpy as np
import pandas as pd
from sklearn_tools.sample import AdaptiveBootstrap


class TestAdaptiveBootstrap(unittest.TestCase):
    def setUp(self):
        pass

    def test_sampling(self):
        y = np.array([0] * 10 + [1] * 90)
        bootstrap = AdaptiveBootstrap(y, n_iter=1)
        for train_idx, test_idx in bootstrap:
            self.assertGreater(y[train_idx].mean(), 0.2)
            self.assertLess(y[train_idx].mean(), 0.8)
            self.assertGreater(y[test_idx].mean(), 0.2)
            self.assertLess(y[test_idx].mean(), 0.8)
