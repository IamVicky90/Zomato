from sklearn.model_selection import GridSearchCV # importing GridSearchCVfor hyperparameter
from sklearn.model_selection import RandomizedSearchCV # importing RandomizedSearchCV hyperparameter
from src.logger import logger
from src.Training_Service.metrics import metrics
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
class hyperparameter:
    def __init__(self):
        self.log=logger.log()
        self.metrics_obj=metrics.measure_metrics()
    def compute_xgboost_hyperparameters(self,x_train,y_train,x_test,y_test):
        xg=XGBRegressor()
        self.log.log_writer('xgboost_hyperparameter_training start','hyperparameter_training.log')
    # when use hyperthread, xgboost may become slower
        parameters = {                        # Parameters that we want to pass
                'num_boost_round': [10, 25, 50],
                'eta': [0.05, 0.1, 0.3],
                'max_depth': [3, 4, 5],
                'subsample': [0.9, 1.0],
                'colsample_bytree': [0.9, 1.0],
            }
        cv=5
        verbose=3
        grid=GridSearchCV(estimator=xg,param_grid=parameters,cv=cv,verbose=verbose) # initialize the GridSearchCV
        self.log.log_writer(f'XGBOOST Parameters for training: {parameters}, with cv = {cv}, verbose= {verbose}','hyperparameter_training.log')
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
        cv=5
        verbose=3
        random=RandomizedSearchCV(estimator=xg,param_distributions=parameters,cv=cv,verbose=verbose) # initialize the RandomizedSearchCV
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

    def compute_random_forest_hyperparameters(self,x_train,y_train,x_test,y_test):
        rf=RandomForestRegressor()
        self.log.log_writer('Random_Forest_hyperparameter_training start','hyperparameter_training.log')
        # # Create the params
        parameters = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            }
        cv=5
        verbose=3
        grid=GridSearchCV(estimator=rf,param_grid=parameters,cv=cv,verbose=verbose) # initialize the GridSearchCV
        self.log.log_writer(f'Random Forest Parameters for training: {parameters}, with cv = {cv}, verbose= {verbose}','hyperparameter_training.log')
        try:
            grid.fit(x_train.to_numpy(),y_train.to_numpy()) # fitting the GridSearchCV with x_train and y_train
            print('@@inside compute_random_forest_hyperparameters')
            self.log.log_writer(f'fitting to GridSearchCV of Random Forest done successfully','hyperparameter_training.log')
        except Exception as e:
            self.log.log_writer(f'fitting to GridSearchCV of Random Forest could not done, error: {str(e)}','hyperparameter_training.log','Error')

        y_pred=grid.predict(x_test.to_numpy()) # predict the x_test
        self.metrics_obj.compute_score(y_test.to_numpy(),y_pred) # computing the r2_score,mean_absolute_error and mean_squared_error
        r2_score_grid=self.metrics_obj.compute_r2_score(y_test.to_numpy(),y_pred)
        n_estimators_grid = grid.best_params_['n_estimators']
        max_features_grid = grid.best_params_['max_features']
        max_depth_grid = grid.best_params_['max_depth']
        bootstrap_grid = grid.best_params_['bootstrap']
        
        random=RandomizedSearchCV(estimator=rf,param_distributions=parameters,cv=5,verbose=3) # initialize the RandomizedSearchCV
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
        bootstrap_random = random.best_params_['bootstrap']
        if r2_score_grid>r2_score_random:
            self.log.log_writer(f'Final params for Random Forest are {grid.get_params()} by GridSearchCV','hyperparameter_training.log')
            return RandomForestRegressor(n_estimators=n_estimators_grid,max_features=max_features_grid,max_depth=max_depth_grid,bootstrap=bootstrap_grid)
        else:
            self.log.log_writer(f'Final params for Random Forest are {random.get_params()} by RandomizedSearchCV','hyperparameter_training.log')
            return RandomForestRegressor(n_estimators=n_estimators_random,max_features=max_features_random,max_depth_random=max_depth_random,bootstrap=bootstrap_random)