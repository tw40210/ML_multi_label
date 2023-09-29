from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
class Evaluator:
    def __init__(self):
        pass

    def get_binary_accuracy(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)

        print('acc:', accuracy)
        print('mean:', np.mean(y_test))
        return accuracy
