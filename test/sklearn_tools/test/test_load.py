import os
import unittest
from sklearn_tools.load import BatchLoader

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

class TestBatchLoader(unittest.TestCase):
    def setUp(self):
        pass

    def test_loading(self):
        loader = BatchLoader(os.path.join(__location__, 'data.csv'), batch_size=5)
        cnt = 0
        for i, df in enumerate(loader):
            self.assertLess(i, 3, 'Too many iterations')
            self.assertGreaterEqual(len(df), 4)
            self.assertEquals(10, len(df.columns))
            cnt += 1
        self.assertEquals(cnt, 2)