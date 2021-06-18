from sklearn.model_selection import GridSearchCV # importing GridSearchCVfor hyperparameter
from sklearn.model_selection import RandomizedSearchCV # importing RandomizedSearchCV hyperparameter
from src.logger import logger
from src.Training_Service.metrics import metrics
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import yaml
class hyperparameter:
    def __init__(self):
        self.log=logger.log()
        self.metrics_obj=metrics.measure_metrics()
    def compute_xgboost_hyperparameters(self,x_train,y_train,x_test,y_test,parameters_grid,cv_grid,verbose_grid,parameters_random,cv_random,verbose_random):
        xg=XGBRegressor()
        self.log.log_writer('xgboost_hyperparameter_training start','hyperparameter_training.log')
        grid=GridSearchCV(estimator=xg,param_grid=parameters_grid,cv=cv_grid,verbose=verbose_grid) # initialize the GridSearchCV
        self.log.log_writer(f'XGBOOST Parameters for training: {parameters_grid}, with cv = {cv_grid}, verbose= {verbose_grid} by GridSearchCV','hyperparameter_training.log')
        try:
            grid.fit(x_train.to_numpy(),y_train.to_numpy()) # fitting the GridSearchCV with x_train and y_train
            self.log.log_writer(f'fitting to GridSearchCV done successfully','hyperparameter_training.log')
        except Exception as e:
            self.log.log_writer(f'fitting to GridSearchCV could not done, error: {str(e)}','hyperparameter_training.log','Error')

        y_pred=grid.predict(x_test.to_numpy()) # predict the x_test
        self.metrics_obj.compute_score(y_test.to_numpy(),y_pred) # computing the r2_score,mean_absolute_error and mean_squared_error
        r2_score_grid=self.metrics_obj.compute_r2_score(y_test.to_numpy(),y_pred)
        num_boost_round_grid = grid.best_params_['num_boost_round']
        eta_grid = grid.best_params_['eta']
        subsample_grid = grid.best_params_['subsample']
        max_depth_grid = grid.best_params_['max_depth']
        colsample_bytree_grid = grid.best_params_['colsample_bytree']
        random=RandomizedSearchCV(estimator=xg,param_distributions=parameters_random,cv=cv_random,verbose=verbose_random) # initialize the RandomizedSearchCV
        self.log.log_writer(f'XGBOOST Parameters for training: {parameters_random}, with cv = {cv_random}, verbose= {verbose_random} by RandomizedSearchCV','hyperparameter_training.log')
        try:
            random.fit(x_train.to_numpy(),y_train.to_numpy()) # fitting the RandomizedSearchCV with x_train and y_train
            self.log.log_writer(f'xgboost fitting to RandomizedSearchCV done successfully','hyperparameter_training.log')
        except Exception as e:
            self.log.log_writer(f'xgboost fitting to RandomizedSearchCV could not done, error: {str(e)}','hyperparameter_training.log','Error')
        y_pred=random.predict(x_test.to_numpy()) # predict the x_test
        score=self.metrics_obj.compute_score(y_test.to_numpy(),y_pred) # computing the r2_score,mean_absolute_error and mean_squared_error
        self.log.log_writer(f'xgboost compute_score resut is {str(score)}','hyperparameter_training.log')
        r2_score_random=self.metrics_obj.compute_r2_score(y_test.to_numpy(),y_pred)
        num_boost_round_random = random.best_params_['num_boost_round']
        eta_random = random.best_params_['eta']
        subsample_random = random.best_params_['subsample']
        max_depth_random = random.best_params_['max_depth']
        colsample_bytree_random = random.best_params_['colsample_bytree']
        if r2_score_grid>r2_score_random:
            self.log.log_writer(f'Final params for XGBOOST are {grid.get_params()} by GridSearchCV','hyperparameter_training.log')
            return xg(num_boost_round_grid=num_boost_round_grid,eta_grid=eta_grid,subsample_grid=subsample_grid,max_depth_grid=max_depth_grid,colsample_bytree_grid=colsample_bytree_grid)
        else:
            self.log.log_writer(f'Final params for XGBOOST are {random.get_params()} by RandomizedSearchCV','hyperparameter_training.log')
            return xg(num_boost_round=num_boost_round_random,eta=eta_random,subsample=subsample_random,max_depth=max_depth_random,colsample_bytree=colsample_bytree_random)

    def compute_random_forest_hyperparameters(self,x_train,y_train,x_test,y_test,parameters_grid,cv_grid,verbose_grid,parameters_random,cv_random,verbose_random):
        rf=RandomForestRegressor()
        self.log.log_writer('Random_Forest_hyperparameter_training start','hyperparameter_training.log')
        grid=GridSearchCV(estimator=rf,param_grid=parameters_grid,cv=cv_grid,verbose=verbose_grid) # initialize the GridSearchCV
        self.log.log_writer(f'Random Forest Parameters for training: {parameters_grid}, with cv = {cv_grid}, verbose= {verbose_grid} by GridSearchCV','hyperparameter_training.log')
        try:
            grid.fit(x_train.to_numpy(),y_train.to_numpy()) # fitting the GridSearchCV with x_train and y_train
            self.log.log_writer(f'fitting to GridSearchCV of Random Forest done successfully','hyperparameter_training.log')
        except Exception as e:
            self.log.log_writer(f'fitting to GridSearchCV of Random Forest could not done, error: {str(e)}','hyperparameter_training.log','Error')

        y_pred=grid.predict(x_test.to_numpy()) # predict the x_test
        self.metrics_obj.compute_score(y_test.to_numpy(),y_pred) # computing the r2_score,mean_absolute_error and mean_squared_error
        r2_score_grid=self.metrics_obj.compute_r2_score(y_test.to_numpy(),y_pred)
        n_estimators_grid = grid.best_params_['n_estimators']
        max_features_grid = grid.best_params_['max_features']
        max_depth_grid = grid.best_params_['max_depth']
        
        random=RandomizedSearchCV(estimator=rf,param_distributions=parameters_random,cv=cv_random,verbose=verbose_random) # initialize the RandomizedSearchCV
        self.log.log_writer(f'Random Forest Parameters for training: {parameters_random}, with cv = {cv_random}, verbose= {verbose_random} by RandomizedSearchCV','hyperparameter_training.log')
        try:
            random.fit(x_train.to_numpy(),y_train.to_numpy()) # fitting the RandomizedSearchCV with x_train and y_train
            self.log.log_writer(f'Random Forest fitting to RandomizedSearchCV done successfully','hyperparameter_training.log')
        except Exception as e:
            self.log.log_writer(f'Random Forest fitting to RandomizedSearchCV could not done, error: {str(e)}','hyperparameter_training.log','Error')
        y_pred=random.predict(x_test.to_numpy()) # predict the x_test
        score=self.metrics_obj.compute_score(y_test.to_numpy(),y_pred) # computing the r2_score,mean_absolute_error and mean_squared_error
        self.log.log_writer(f'Random Forest compute_score resut is {str(score)}','hyperparameter_training.log')
        r2_score_random=self.metrics_obj.compute_r2_score(y_test.to_numpy(),y_pred)
        n_estimators_random = random.best_params_['n_estimators']
        max_features_random = random.best_params_['max_features']
        max_depth_random = random.best_params_['max_depth']
        if r2_score_grid>r2_score_random:
            self.log.log_writer(f'Final params for Random Forest are {grid.get_params()} by GridSearchCV','hyperparameter_training.log')
            return RandomForestRegressor(n_estimators=n_estimators_grid,max_features=max_features_grid,max_depth=max_depth_grid)
        else:
            self.log.log_writer(f'Final params for Random Forest are {random.get_params()} by RandomizedSearchCV','hyperparameter_training.log')
            return RandomForestRegressor(n_estimators=n_estimators_random,max_features=max_features_random,max_depth=max_depth_random)