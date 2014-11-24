import os
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn_tools.misc import which
from sklearn.cross_validation import train_test_split


XGBOOST_HOME = os.getenv('XGBOOST_HOME')

if not XGBOOST_HOME:
    xgboost_executable = which('xgboost')
    if xgboost_executable:
        XGBOOST_HOME = os.path.dirname(xgboost_executable)
    else:
        raise RuntimeError('XGBOOST_HOME is not set, and xgboost is not in PATH')

sys.path.append(os.path.join(XGBOOST_HOME, 'wrapper'))
import xgboost as xb


class XGBoostClassifier(BaseEstimator, TransformerMixin):
    """
    TODO: write some docs here
    """
    def __init__(self,
                 n_jobs=1,
                 model=None,
                 txt_model=None, txt_model_feature_map=None,
                 n_iter=10,
                 params=None,
                 n_class=None,
                 plist=None,
                 missing=-1,
                 validation_percent=0.1
                 ):
        super(XGBoostClassifier, self).__init__()
        self.n_jobs = n_jobs
        self.model = model
        self.txt_model = txt_model
        self.txt_model_feature_map = txt_model_feature_map
        self.params = params if params else {}
        self.params['objective'] = 'multi:softprob'
        self.n_class = n_class
        self.plist = plist if plist else []
        self.n_iter = n_iter
        self.missing = missing
        self.validation_percent = validation_percent
        self.booster = None
        self.enc = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        self.enc = self.enc.fit(y)
        y = self.enc.transform(y)
        self.classes_ = self.enc.classes_
        if sample_weight is not None:
            X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val = train_test_split(X, y, sample_weight, test_size=self.validation_percent)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_percent)
            sample_weight_train, sample_weight_val = None, None
        dtrain = xb.DMatrix(X_train, label=y_train, missing=self.missing, weight=sample_weight_train)
        dtest = xb.DMatrix(X_val, label=y_val, missing=self.missing, weight=sample_weight_val)
        self.params['num_class'] = self.n_class
        self.params['nthread'] = self.n_jobs
        self.booster = xb.train(self.params, dtrain, self.n_iter, [(dtest, 'eval')])
        if self.model:
            self.booster.save_model(self.model)
        if self.txt_model:
            self.booster.dump_model(self.txt_model, self.txt_model_feature_map)
        return self

    def transform(self, X):
        dtest = xb.DMatrix(X, missing=self.missing)
        return self.booster.predict(dtest).reshape((X.shape[0], len(self.classes_)))

    def predict_proba(self, X):
        return self.transform(X)