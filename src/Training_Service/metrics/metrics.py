from src.logger import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class measure_metrics:
    def __init__(self):
        self.log = logger.log()

    def compute_r2_score(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def compute_mean_squared_error(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def compute_mean_absolute_error(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def compute_score(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        result = f"r2_score is: {r2}, mean squared error is {mse},mean absolute error is {mae}"
        return result
