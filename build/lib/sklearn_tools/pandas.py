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
    def __init__(self, column, transformer):
        self.column = column
        self.transformer = transformer

    def fit(self, X, y=None):
        if y is None:
            self.transformer = self.transformer.fit(X[self.column].values)
        else:
            self.transformer = self.transformer.fit(X[self.column].values, y)

    def transform(self, X):
        X[self.column] = self.transformer.transform(X[self.column].values)