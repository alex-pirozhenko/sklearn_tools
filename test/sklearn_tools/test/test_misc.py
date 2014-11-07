import unittest
from sklearn_tools.misc import JSONToObjectTransformer
import numpy as np
import pandas as pd


class TestJSONToObjectTransformer(unittest.TestCase):
    def setUp(self):
        pass

    def test_list_of_strings(self):
        transformer = JSONToObjectTransformer()
        result = transformer.fit_transform(['{"123":10,"234":20}', '{"1234":100,"2345":200}', ])
        self.assertEquals(list(result), [{'123': 10, '234': 20}, {'1234': 100, '2345': 200}])

    def test_ndarray_of_strings(self):
        transformer = JSONToObjectTransformer()
        result = transformer.fit_transform(np.array(['{"123":10,"234":20}', '{"1234":100,"2345":200}', ]))
        self.assertEquals(list(result), [{'123': 10, '234': 20}, {'1234': 100, '2345': 200}])

    def test_series_of_strings(self):
        transformer = JSONToObjectTransformer()
        result = transformer.fit_transform(pd.Series(['{"123":10,"234":20}', '{"1234":100,"2345":200}', ]))
        self.assertEquals(list(result), [{'123': 10, '234': 20}, {'1234': 100, '2345': 200}])