from itertools import imap
import json
from sklearn.base import TransformerMixin, BaseEstimator
from functools import partial
import scipy as sp
import scipy.sparse
import numpy as np


class NoOpTransformer(object):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class JSONToObjectTransformer(NoOpTransformer, TransformerMixin):
    """
    Assuming that X is an iterable over strings with JSON, this transformer parses the data
    """
    def __init__(self, lazy=True, **kwargs):
        """
        :param kwargs: parameters for json.loads
        :return:
        """
        self.kwargs = kwargs
        self.func = partial(imap if lazy else map, partial(json.loads, **self.kwargs))

    def transform(self, X):
        return self.func(X)


class RowNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        normalizer = sp.sparse.lil_matrix((X.shape[0], X.shape[0]))
        normalizer.setdiag(1.0/X.sum(1))
        return normalizer * X


class GBRTInitialEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, est):
        self.est = est

    def predict(self, X):
        return self.est.predict_proba(X)[:, 1][:, np.newaxis]

    def fit(self, X, y):
        self.est.fit(X, y)