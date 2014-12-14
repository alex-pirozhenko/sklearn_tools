import cPickle
import time
import wabbit_wappa as ww
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(0 - x))


@np.vectorize
def field_formatter(value, col_name='', sep=' '):
    if value != '' and value is not None:
        return col_name + sep + value + ' '
    else:
        return ' '


class DataFrameToVWTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_columns, dump_to=None, namespaces=None):
        super(DataFrameToVWTransformer, self).__init__()
        self.categorical_columns = categorical_columns
        self.dump_to = dump_to
        self.namespaces = namespaces

    def fit(self, X, y=None, tag=None):
        """
        Remember y
        """
        self.y = y
        self.tag = tag
        return self

    def transform(self, X):
        """
        :type X: pd.DataFrame
        """
        X = X.copy(deep=False)
        columns = list(X.columns)
        X['__res'] = ''
        if not (self.y is None):
            X['__res'] += self.y.astype(str)
            X['__res'] += ' '
        if not (self.tag is None):
            X['__res'] += self.tag.astype(str)
        if self.namespaces:
            for name, column_names in self.namespaces.items():
                if column_names:
                    X['__res'] += '|%s ' % name
                    for c in column_names:
                        if c in self.categorical_columns:
                            X.loc[X[c].notnull(), '__res'] += field_formatter(X[c][X[c].notnull()].astype(str), col_name=c, sep='_')
                        else:
                            X.loc[X[c].notnull(), '__res'] += field_formatter(X[c][X[c].notnull()].astype(str), col_name=c, sep=':')
                        columns.remove(c)
        if [_ for _ in self.categorical_columns if _ in columns]:
            X['__res'] += '|c '
            for c in [_ for _ in self.categorical_columns if _ in columns]:
                X.loc[X[c].notnull(), '__res'] += field_formatter(X[c][X[c].notnull()].astype(str), col_name=c, sep='_')
                columns.remove(c)
        if columns:
            X['__res'] += '|i '
            for c in columns:
                X['__res'] += field_formatter(X[c].astype(str), col_name=c, sep=':')
                columns.remove(c)
        X = X[['__res']]
        if self.dump_to:
            X.to_csv(self.dump_to, header=False, index=False)
        return X['__res']




class VowpalWabbitEstimator(BaseEstimator):
    def __init__(self, vw_builder):
        self.vw = vw_builder()
        assert isinstance(self.vw, ww.VW)

    def fit(self, X, y):
        if isinstance(pd.DataFrame, X):
            return self._fit_data_frame(X, y)
        raise RuntimeError("Unsupported data type")

    def _fit_data_frame(self, X, y):
        vw = self.vw
        assert isinstance(vw, VW)
        assert isinstance(X, pd.DataFrame)
        for r, label in zip(X.iterrows(), y):
            vw.send_example(
                response=label,
                features=list(r[1].iterkv)
            )



if __name__ == '__main__':
    with open("train_prepaired.pkl", "r") as f:
        train = cPickle.load(f)

    trainLabels = pd.read_csv("trainLabels", true_values=['YES'], false_values=['NO', ''], na_filter=False)
    trainId = trainLabels["id"]

    del trainLabels["id"]

    trainLabels = trainLabels.replace(0, -1)
    rand = np.random.RandomState(1)
    nrows = trainLabels.shape[0]
    trainIdx = rand.uniform(0, 1, nrows) < 0.8
    xCol = train.columns[:135].values
    cCol = train.columns[135:].values
    test = train[~trainIdx]
    testLabels = trainLabels[~trainIdx]
    train = train[trainIdx]
    trainLabels = trainLabels[trainIdx]

    vw_train = []
    for idx, row in train.iterrows():
        vw_train.append([(i, round(v, 3)) if v != 1.0 else i for i, v in row[row > 0].iteritems()])
    del train
    vw_test = []
    for idx, row in test.iterrows():
        vw_test.append([(i, round(v, 3)) if v != 1.0 else i for i, v in row[row > 0].iteritems()])
    del test

    logLossTest = pd.Series()
    for label in trainLabels.columns[11:]:
        start = time.time()
        if label == "y14":
            logLossTest[label] = 0.0
        else:
            vw = ww.VW(loss_function='logistic', b=31, f="vw_model.model")

            for idx, row in enumerate(vw_train):
                lb = trainLabels[label].iloc[idx]
                vw.send_example(response=lb, features=row)
                # print "send"
            # test & log loss
            r = 0.0
            for idx, row in enumerate(vw_test):
                response = vw.get_prediction(features=row)
                p = sigmoid(response.prediction)
                a = (testLabels[label].iloc[idx] > 0).astype(int)
                if p < 1e-15:
                    p = 1e-15
                if p > (1 - 1e-15):
                    p = 1 - 1e-15
                r = r - a * np.log(p) - (1 - a) * np.log(1 - p)

            r /= float(test.shape[0])
            logLossTest[label] = r
            print round((time.time() - start)/60, 2), label, r
            vw.close()
