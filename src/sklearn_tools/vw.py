import cPickle
import time
import wabbit_wappa as ww
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(0 - x))


class DataFrameToVWTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_columns, tag_column=None, nan_value=-1, dump_to=None):
        super(DataFrameToVWTransformer, self).__init__()
        self.categorical_columns = categorical_columns
        self.tag_column = tag_column
        self.nan_value = nan_value
        self.dump_to = dump_to

    def fit(self, X, y=None):
        """
        Remember y
        """
        self.y = y
        return self

    def transform(self, X):
        """
        :type X: pd.DataFrame
        """
        X = X.copy(deep=False)
        columns = X.columns
        if not (self.y is None):
            X['__label'] = self.y
            use_y = True
        else:
            use_y = False
        X['__res'] = ''
        if not (self.y is None):
            X['__res'] += X['__label']
            X['__res'] += ' '
        X['__res'] += ('' if not self.tag_column else X[self.tag_column].values.astype(str))
        X['__res'] += '|c '
        for c in self.categorical_columns:
            X['__res'] += c + '_'
            X['__res'] += ('' if not self.tag_column else X[self.tag_column].values.astype(str))
            X['__res'] += ' '
        X['__res'] += '|i '
        for c in columns:
            if c not in self.categorical_columns and c != self.tag_column:
                X['__res'] += c + ':'
                X['__res'] += ('' if not self.tag_column else X[self.tag_column].values.astype(str))
                X['__res'] += ' '

        X = X[['__res']]

        if self.dump_to:
            X.to_csv(self.dump_to, header=False, index=False)
        return X['__res'].values




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
