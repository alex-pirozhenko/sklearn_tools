from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameToMatrixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        for c in self.columns:
            if c not in X.columns:
                raise RuntimeError('Unknown column : "' + c + '"')
        return self

    def transform(self, X):
        return X[self.columns].as_matrix()


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, selector, transformer):
        self.selector = selector
        self.transformer = transformer

    def fit(self, X, y=None):
        if y is None:
            self.transformer = self.transformer.fit(X[self.selector].values)
        else:
            self.transformer = self.transformer.fit(X[self.selector].values, y)

    def transform(self, X):
        X[self.selector] = self.transformer.transform(X[self.selector].values)