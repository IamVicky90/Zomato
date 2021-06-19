from src.logger import logger
import pickle
import os
from src.Training_Service.metrics import metrics
from src.Training_Service.Hyperparameter import hyperparameter_tunning
import os
class model_operations:
    def __init__(self):
        self.log=logger.log()
        self.metrics_obj=metrics.measure_metrics()
        self.tunner=hyperparameter_tunning.hyperparameter()
    def train_model_with_clusters_with_hyperparameter_tuning(self,x,y,x_test,y_test,parameters_random_forest_grid,cv_random_forest_grid,verbose_random_forest_grid,parameters_random_forest_random,cv_random_forest_random,verbose_random_forest_random,
    parameters_xgboost_grid,cv_xgboost_grid,verbose_xgboost_grid,parameters_xgboost_random,cv_xgboost_random,verbose_xgboost_random):
        '''From EDA in Jupyter Notebook we see that RandomForestRegressor and XGBRegressor are predicting well so we use only these two models for the prediction and hyperparameters'''
        self.log.log_writer('Model Operations started!','model_operations.log')
        for cluster_no in x['cluster'].unique():
            cluster_x_Train=x[x['cluster']==cluster_no]
            cluster_y_Train=y[cluster_x_Train.index]
            cluster_x_test=x_test[x_test['cluster']==cluster_no]
            cluster_y_test=y_test[cluster_x_test.index]
            try:
                self.rf=self.tunner.compute_random_forest_hyperparameters(cluster_x_Train,cluster_y_Train,cluster_x_test,cluster_y_test,parameters_random_forest_grid,cv_random_forest_grid,verbose_random_forest_grid,parameters_random_forest_random,cv_random_forest_random,verbose_random_forest_random)
                # self.rf=RandomForestRegressor()
                self.rf.fit(cluster_x_Train.drop(['cluster'],axis=1),cluster_y_Train)
                model_name_reandom_forest= f'random_forest_regressor_cluster_no_{cluster_no}.sav'
                path_random=os.path.join(os.getcwd(),'models',model_name_reandom_forest)
                pickle.dump(self.rf, open(path_random, 'wb'))
                self.log.log_writer(f'Successfully saved the {model_name_reandom_forest} at path {path_random}','model_operations.log')
            except Exception as e:
                try:
                    self.log.log_writer(f'Could not saved the {model_name_reandom_forest} at path {path_random} error: {str(e)}','model_operations.log','ERROR')
                except Exception as NameError:
                    self.log.log_writer(f'NameError occured in train_model_with_clusters','model_operations.log','ERROR')
            try:
                self.xg_model=self.rn=self.tunner.compute_xgboost_hyperparameters(cluster_x_Train,cluster_y_Train,cluster_x_test,cluster_y_test,parameters_xgboost_grid,cv_xgboost_grid,verbose_xgboost_grid,parameters_xgboost_random,cv_xgboost_random,verbose_xgboost_random)  
                # self.xg_model=XGBRegressor() 
                self.xg_model.fit(cluster_x_Train.drop(['cluster'],axis=1).to_numpy(),cluster_y_Train.to_numpy())
                model_name_xgboost= f'XGBOOST_Regressor_cluster_no_{cluster_no}.sav'
                path_xgboost=os.path.join(os.getcwd(),'models',model_name_xgboost)
                pickle.dump(self.xg_model, open(path_xgboost, 'wb'))
                self.log.log_writer(f'Successfully saved the {model_name_xgboost} at path {path_xgboost}','model_operations.log')
            except Exception as e:
                try:
                    self.log.log_writer(f'Could not saved the {model_name_xgboost} at path {path_xgboost} error: {str(e)}','model_operations.log','ERROR')
                except Exception as NameError:
                    self.log.log_writer(f'NameError occured in train_model_with_clusters','model_operations.log','ERROR')
    def selct_best_model_with_cluster(self,x_test,y_test):
        for cluster_no in x_test['cluster'].unique():
            cluster_x=x_test[x_test['cluster']==cluster_no]
            cluster_y=y_test[cluster_x.index]
            path=os.path.join(os.getcwd(),'models')
            for file in os.listdir(path):
                if str(cluster_no) in file and '.sav' in file:
                    if 'random_forest' in file:
                        try:
                            model_random_forest=pickle.load(open(os.path.join(path,file),'rb'))
                            predict_random_forest=model_random_forest.predict(cluster_x.drop(['cluster'],axis=1).to_numpy())
                            
                            random_forest_file=file
                            self.log.log_writer(f'Sucessfully load predict the model {random_forest_file}','model_operations.log')
                        except Exception as e:
                            self.log.log_writer(f'Couldnot load predict the model {random_forest_file} error {str(e)}','model_operations.log','Error')
                    elif 'XGBOOST_Regressor' in file:
                        try:
                            model_XGBOOST_Regressor=pickle.load(open(os.path.join(path,file),'rb'))
                            predict_model_XGBOOST_Regressor=model_XGBOOST_Regressor.predict(cluster_x.drop(['cluster'],axis=1).to_numpy())
                            XGBOOST_Regressor_file=file
                            self.log.log_writer(f'Sucessfully load predict the model {XGBOOST_Regressor_file}','model_operations.log')
                        except Exception as e:
                            self.log.log_writer(f'Couldnot load predict the model {XGBOOST_Regressor_file} error {str(e)}','model_operations.log','Error')
            random_forest_r2_score=self.metrics_obj.compute_r2_score(y_true=cluster_y,y_pred=predict_random_forest)
            XGBOOST_Regressor_metrics_r2_score=self.metrics_obj.compute_r2_score(y_true=cluster_y,y_pred=predict_model_XGBOOST_Regressor)
            self.log.log_writer(f'The r2_score for random_forest is {random_forest_r2_score} and xgboost regressor is {XGBOOST_Regressor_metrics_r2_score} for cluster no {cluster_no}','model_operations.log')

            if XGBOOST_Regressor_metrics_r2_score>random_forest_r2_score:
                try:
                    os.remove(os.path.join(path,random_forest_file))
                    self.log.log_writer(f'Sucessfully, remove the file {random_forest_file} because it has low r2_score as compared to xgboost','model_operations.log','Warning')
                except Exception as e:
                    self.log.log_writer(f'Couldnot remove the file {random_forest_file} error {str(e)}','model_operations.log','Error')
            else:
                try:
                    os.remove(os.path.join(path,XGBOOST_Regressor_file))
                    self.log.log_writer(f'Sucessfully, remove the file {XGBOOST_Regressor_file} because it has low r2_score as compared to random forest','model_operations.log','Warning')
                except Exception as e:
                    self.log.log_writer(f'Couldnot remove the file {XGBOOST_Regressor_file} error {str(e)}','model_operations.log','Error')
        