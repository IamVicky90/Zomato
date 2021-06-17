from src.logger import logger
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
class measure_metrics:
    def __init__(self):
        self.log=logger.log()
    def compute_r2_score(self,y_true,y_pred):
        return r2_score(y_true,y_pred)
        