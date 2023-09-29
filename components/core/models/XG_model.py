from core.models.basic_models import BasicModel
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

class XGBoostModel(BasicModel):
    def __init__(self, multi_output:bool = True):
        super(XGBoostModel, self).__init__()
        self.model = None
        self.multi_output = multi_output
    def train_multi_label(self, X_train, X_test, y_train, y_test):
        self.model=[]

        for i in range(y_test.shape[1]):
            y_train_single = y_train[y_train.columns[i]]
            y_test_single = y_test[y_test.columns[i]]

            dtrain = xgb.DMatrix(X_train, label=y_train_single)
            dtest = xgb.DMatrix(X_test, label=y_test_single)
            params = {
                # 'objective': 'binary:logistic',
                'num_class': 2,
                'eta': 0.1,
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
                "max_depth": 10,
                "verbose": 0
            }

            self.model.append(xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=50,
                evals=[(dtrain, "train"), (dtest, "test")],
                verbose_eval=0
            ))

        return


    def predict_multi_label(self, data:np.ndarray) -> np.ndarray:
        result=[]
        for i in range(len(self.model)):
            dtest = xgb.DMatrix(data)
            # Make predictions
            y_pred = self.model[i].predict(dtest, iteration_range=(10,20))

            result.append(y_pred)
        result = np.vstack(result).T
        return result


    def train(self, X_train, X_test, y_train, y_test):
        if self.multi_output:
            return self.train_multi_label(X_train, X_test, y_train, y_test)
        else:
            return
    def predict(self, data: np.ndarray) ->np.ndarray:
        if self.multi_output:
            return self.predict_multi_label(data)
        else:
            return

