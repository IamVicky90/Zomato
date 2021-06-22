import json
# importing GridSearchCVfor hyperparameter
from sklearn.model_selection import GridSearchCV
# importing RandomizedSearchCV hyperparameter
from sklearn.model_selection import RandomizedSearchCV
from src.logger import logger
from src.Training_Service.metrics import metrics
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import os


class hyperparameter:
    def __init__(self):
        self.log = logger.log()
        self.metrics_obj = metrics.measure_metrics()

    def compute_xgboost_hyperparameters(self, x_train, y_train, x_test, y_test, parameters_grid, cv_grid, verbose_grid, parameters_random, cv_random, verbose_random, cluster_no):
        xg = XGBRegressor()
        self.log.log_writer(
            'xgboost_hyperparameter_training start', 'hyperparameter_training.log')
        grid = GridSearchCV(estimator=xg, param_grid=parameters_grid,
                            cv=cv_grid, verbose=verbose_grid)  # initialize the GridSearchCV
        self.log.log_writer(
            f'XGBOOST Parameters for training: {parameters_grid}, with cv = {cv_grid}, verbose= {verbose_grid} by GridSearchCV', 'hyperparameter_training.log')
        try:
            # fitting the GridSearchCV with x_train and y_train
            grid.fit(x_train.drop(
                ['cluster'], axis=1).to_numpy(), y_train.to_numpy())
            self.log.log_writer(
                f'fitting to GridSearchCV done successfully', 'hyperparameter_training.log')
        except Exception as e:
            self.log.log_writer(
                f'fitting to GridSearchCV could not done, error: {str(e)}', 'hyperparameter_training.log', 'Error')

        y_pred = grid.predict(x_test.drop(
            ['cluster'], axis=1).to_numpy())  # predict the x_test
        # computing the r2_score,mean_absolute_error and mean_squared_error
        self.metrics_obj.compute_score(y_test.to_numpy(), y_pred)
        r2_score_grid = self.metrics_obj.compute_r2_score(
            y_test.to_numpy(), y_pred)
        num_boost_round_grid = grid.best_params_['num_boost_round']
        eta_grid = grid.best_params_['eta']
        subsample_grid = grid.best_params_['subsample']
        max_depth_grid = grid.best_params_['max_depth']
        colsample_bytree_grid = grid.best_params_['colsample_bytree']
        random = RandomizedSearchCV(estimator=xg, param_distributions=parameters_random,
                                    cv=cv_random, verbose=verbose_random)  # initialize the RandomizedSearchCV
        self.log.log_writer(
            f'XGBOOST Parameters for training: {parameters_random}, with cv = {cv_random}, verbose= {verbose_random} by RandomizedSearchCV', 'hyperparameter_training.log')
        try:
            # fitting the RandomizedSearchCV with x_train and y_train
            random.fit(x_train.drop(
                ['cluster'], axis=1).to_numpy(), y_train.to_numpy())
            self.log.log_writer(
                f'xgboost fitting to RandomizedSearchCV done successfully', 'hyperparameter_training.log')
        except Exception as e:
            self.log.log_writer(
                f'xgboost fitting to RandomizedSearchCV could not done, error: {str(e)}', 'hyperparameter_training.log', 'Error')
        y_pred = random.predict(x_test.drop(
            ['cluster'], axis=1).to_numpy())  # predict the x_test
        # computing the r2_score,mean_absolute_error and mean_squared_error
        score = self.metrics_obj.compute_score(y_test.to_numpy(), y_pred)
        self.log.log_writer(
            f'xgboost compute_score resut is {str(score)}', 'hyperparameter_training.log')
        r2_score_random = self.metrics_obj.compute_r2_score(
            y_test.to_numpy(), y_pred)
        num_boost_round_random = random.best_params_['num_boost_round']
        eta_random = random.best_params_['eta']
        subsample_random = random.best_params_['subsample']
        max_depth_random = random.best_params_['max_depth']
        colsample_bytree_random = random.best_params_['colsample_bytree']
        if r2_score_grid > r2_score_random:
            self.log.log_writer(
                f'Final params for XGBOOST are {grid.get_params()} by GridSearchCV', 'hyperparameter_training.log')
            with open(os.path.join('reports/params.json'), 'r+') as f:
                params = {f'Parameters Selected for XGBOOST for cluster {cluster_no} by GridSearchCV are': str(
                    grid.get_params())}
                data = json.load(f)
                data.update(params)
                f.seek(0)
                json.dump(data, f, indent=4)
            return XGBRegressor(num_boost_round_grid=num_boost_round_grid, eta_grid=eta_grid, subsample_grid=subsample_grid, max_depth_grid=max_depth_grid, colsample_bytree_grid=colsample_bytree_grid)
        else:

            self.log.log_writer(
                f'Final params for XGBOOST are {random.get_params()} by RandomizedSearchCV', 'hyperparameter_training.log')
            with open(os.path.join('reports/params.json'), 'r+') as f:
                params = {
                    f'Parameters Selected for XGBOOST for cluster {cluster_no} by RandomizedSearchCV are': str(random.get_params())
                }
                data = json.load(f)
                data.update(params)
                f.seek(0)
                json.dump(data, f, indent=4)
            return XGBRegressor(num_boost_round=num_boost_round_random, eta=eta_random, subsample=subsample_random, max_depth=max_depth_random, colsample_bytree=colsample_bytree_random)

    def compute_random_forest_hyperparameters(self, x_train, y_train, x_test, y_test, parameters_grid, cv_grid, verbose_grid, parameters_random, cv_random, verbose_random, cluster_no):
        rf = RandomForestRegressor()
        self.log.log_writer(
            'Random_Forest_hyperparameter_training start', 'hyperparameter_training.log')
        grid = GridSearchCV(estimator=rf, param_grid=parameters_grid,
                            cv=cv_grid, verbose=verbose_grid)  # initialize the GridSearchCV
        self.log.log_writer(
            f'Random Forest Parameters for training: {parameters_grid}, with cv = {cv_grid}, verbose= {verbose_grid} by GridSearchCV', 'hyperparameter_training.log')
        try:
            # fitting the GridSearchCV with x_train and y_train
            grid.fit(x_train.drop(
                ['cluster'], axis=1).to_numpy(), y_train.to_numpy())
            self.log.log_writer(
                f'fitting to GridSearchCV of Random Forest done successfully', 'hyperparameter_training.log')
        except Exception as e:
            self.log.log_writer(
                f'fitting to GridSearchCV of Random Forest could not done, error: {str(e)}', 'hyperparameter_training.log', 'Error')

        y_pred = grid.predict(x_test.drop(
            ['cluster'], axis=1).to_numpy())  # predict the x_test
        # computing the r2_score,mean_absolute_error and mean_squared_error
        self.metrics_obj.compute_score(y_test.to_numpy(), y_pred)
        r2_score_grid = self.metrics_obj.compute_r2_score(
            y_test.to_numpy(), y_pred)
        n_estimators_grid = grid.best_params_['n_estimators']
        max_features_grid = grid.best_params_['max_features']
        max_depth_grid = grid.best_params_['max_depth']

        random = RandomizedSearchCV(estimator=rf, param_distributions=parameters_random,
                                    cv=cv_random, verbose=verbose_random)  # initialize the RandomizedSearchCV
        self.log.log_writer(
            f'Random Forest Parameters for training: {parameters_random}, with cv = {cv_random}, verbose= {verbose_random} by RandomizedSearchCV', 'hyperparameter_training.log')
        try:
            # fitting the RandomizedSearchCV with x_train and y_train
            random.fit(x_train.drop(
                ['cluster'], axis=1).to_numpy(), y_train.to_numpy())
            self.log.log_writer(
                f'Random Forest fitting to RandomizedSearchCV done successfully', 'hyperparameter_training.log')
        except Exception as e:
            self.log.log_writer(
                f'Random Forest fitting to RandomizedSearchCV could not done, error: {str(e)}', 'hyperparameter_training.log', 'Error')
        y_pred = random.predict(x_test.drop(
            ['cluster'], axis=1).to_numpy())  # predict the x_test
        # computing the r2_score,mean_absolute_error and mean_squared_error
        score = self.metrics_obj.compute_score(y_test.to_numpy(), y_pred)
        self.log.log_writer(
            f'Random Forest compute_score resut is {str(score)}', 'hyperparameter_training.log')
        r2_score_random = self.metrics_obj.compute_r2_score(
            y_test.to_numpy(), y_pred)
        n_estimators_random = random.best_params_['n_estimators']
        max_features_random = random.best_params_['max_features']
        max_depth_random = random.best_params_['max_depth']
        if r2_score_grid > r2_score_random:
            self.log.log_writer(
                f'Final params for Random Forest are {grid.get_params()} by GridSearchCV', 'hyperparameter_training.log')
            with open(os.path.join('reports/params.json'), 'r+') as f:
                params = {
                    f'Parameters Selected for Random Forest for cluster {cluster_no} by GridSearchCV are': str(grid.get_params())
                }
                data = json.load(f)
                data.update(params)
                f.seek(0)
                json.dump(data, f, indent=4)
            return RandomForestRegressor(n_estimators=n_estimators_grid, max_features=max_features_grid, max_depth=max_depth_grid)
        else:
            self.log.log_writer(
                f'Final params for Random Forest are {random.get_params()} by RandomizedSearchCV', 'hyperparameter_training.log')
            with open(os.path.join('reports/params.json'), 'r+') as f:
                params = {
                    f'Parameters Selected for Random Forest for cluster {cluster_no} by RandamizedSearchCV are': str(random.get_params())
                }
                data = json.load(f)
                data.update(params)
                f.seek(0)
                json.dump(data, f, indent=4)
            return RandomForestRegressor(n_estimators=n_estimators_random, max_features=max_features_random, max_depth=max_depth_random)
