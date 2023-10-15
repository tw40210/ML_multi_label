import abc
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict

class BasicModel(abc.ABC):
    def __init__(self):
        self.best_score=defaultdict(OrderedDict)

    @abc.abstractmethod
    def predict(self, data: np.ndarray)->pd.DataFrame:
        pass

    @abc.abstractmethod
    def train(self, params, valid_sets, train_set, callbacks):
        pass