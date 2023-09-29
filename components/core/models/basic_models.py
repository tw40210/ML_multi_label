import abc
import pandas as pd
import numpy as np

class BasicModel(abc.ABC):

    @abc.abstractmethod
    def predict(self, data: np.ndarray)->np.ndarray:
        pass
