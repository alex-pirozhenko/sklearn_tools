from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np
import numpy.random.mtrand
from sklearn.preprocessing import LabelEncoder
from SdA import SdA
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

__all__ = ['StackedDenoisingAutoencoder']

# TODO: l1/l2 regularization
class StackedDenoisingAutoencoder(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self,
                 corruption_level,
                 hidden_layers_sizes=None,
                 random_seed=None,
                 pretraining_learning_rate=0.1,
                 pretraining_n_passes=15,
                 finetune_learning_rate=0.1,
                 finetuning_n_passes=10):
        """
        :type random_seed: int
        :param random_seed: random state

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_level: float
        :param corruption_level: amount of corruption to use for each
                                  layer

        :type finetune_learning_rate: float
        :param finetune_learning_rate: learning rate used in the finetune stage
        (factor for the stochastic gradient)

        :type pretraining_n_passes: int
        :param pretraining_n_passes: number of epoch to do pretraining

        :type pretraining_learning_rate: float
        :param pretraining_learning_rate: learning rate to be used during pre-training

        :type finetuning_n_passes: int
        :parafinetuning_n_passeses: maximal number of iterations ot run the optimizer
        """
        self.corruption_level = corruption_level
        self.hidden_layer_sizes = hidden_layers_sizes
        self.np_rng = np.random.mtrand.RandomState(random_seed)
        self.pretraining_learning_rate = pretraining_learning_rate
        self.pretraining_n_passes = pretraining_n_passes
        self.finetune_learning_rate = finetune_learning_rate
        self.finetuning_n_passes = finetuning_n_passes
        self.sda = None
        self._enc = LabelEncoder()

    def fit(self, X, y):
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        y = np.vstack(
            np.where(y == i, 1, 0)
            for i, _ in enumerate(self.classes_)
        ).T
        sda = SdA(
            input=X,
            label=y,
            n_ins=X.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            n_outs=len(self.classes_),
            numpy_rng=self.np_rng
        )
        sda.pretrain(
            lr=self.pretraining_learning_rate,
            corruption_level=self.corruption_level,
            epochs=self.pretraining_n_passes,
        )
        sda.finetune(lr=self.finetune_learning_rate, epochs=self.finetuning_n_passes)
        self.sda = sda
        return self

    def transform(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        return self.sda.predict(X)