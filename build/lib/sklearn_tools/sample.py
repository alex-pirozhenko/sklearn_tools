from sklearn.cross_validation import Bootstrap
from sklearn.utils import check_random_state
import numpy as np

class AdaptiveBootstrap(Bootstrap):
    """
    Random adaptive sampling with replacement cross-validation iterator
    Makes the class distribution uniform.
    Parameters
    ----------
    n : int
        Total number of elements in the dataset.

    n_iter : int (default is 3)
        Number of bootstrapping iterations

    train_size : int or float (default is 0.5)
        If int, number of samples to include in the training split
        (should be smaller than the total number of samples passed
        in the dataset).

        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.

    test_size : int or float or None (default is None)
        If int, number of samples to include in the training set
        (should be smaller than the total number of samples passed
        in the dataset).

        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split.

        If None, n_test is set as the complement of n_train.

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    """

    def __init__(self, y, n_iter=3, train_size=.5, test_size=None, random_state=None, n_bootstraps=None):
        super(AdaptiveBootstrap, self).__init__(y.shape[0], n_iter, train_size, test_size, random_state, n_bootstraps)
        y = np.array(y)
        self.classes_proba = dict((_, 1.0/(y == _).sum().astype(np.float64)) for _ in np.unique(y))
        self.train_proba = np.vectorize(self.classes_proba.get)(y)

    def __iter__(self):
        rng = check_random_state(self.random_state)
        for i in range(self.n_iter):
            # random partition
            permutation = rng.permutation(self.n)
            ind_train = permutation[:self.train_size]
            ind_test = permutation[self.train_size:self.train_size
                                                   + self.test_size]

            # bootstrap in each split individually
            train = np.random.choice(ind_train, size=(self.train_size,), replace=True,
                                     p=self.train_proba[ind_train]/self.train_proba[ind_train].sum())
            test = np.random.choice(ind_test, size=(self.train_size,), replace=True,
                                    p=self.train_proba[ind_test]/self.train_proba[ind_test].sum())
            yield train, test
