from src.logger import logger
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from xgboost import XGBRegressor
from src.Training_Service.metrics import metrics
import shutil,os
class model_operations:
    def __init__(self):
        self.log=logger.log()
        self.rn=RandomForestRegressor()
        self.xg_model=XGBRegressor()
        self.metrics_obj=metrics.measure_metrics()
    def train_model_with_clusters(self,x,y):
        '''From EDA in Jupyter Notebook we see that RandomForestRegressor and XGBRegressor are predicting well so we use only these two models for the prediction'''
        self.log.log_writer('Model Operations started!','model_operations.log')
        for cluster_no in x['cluster'].unique():
            cluster_x=x[x['cluster']==cluster_no]
            cluster_y=y[cluster_x.index]
            try:
                self.rn.fit(cluster_x.drop(['cluster'],axis=1),cluster_y)
                self.xg_model.fit(cluster_x.to_numpy(),cluster_y.to_numpy())
                model_name_reandom_forest= f'random_forest_regressor_cluster_no_{cluster_no}.sav'
                path_random=os.path.join(os.getcwd(),'models',model_name_reandom_forest)
                pickle.dump(self.rn, open(path_random, 'wb'))
                self.log.log_writer(f'Successfully saved the {model_name_reandom_forest} at path {path_random}','model_operations.log')
            except Exception as e:
                try:
                    self.log.log_writer(f'Could not saved the {model_name_reandom_forest} at path {path_random} error: {str(e)}','model_operations.log','ERROR')
                except Exception as NameError:
                    self.log.log_writer(f'NameError occured in train_model_with_clusters','model_operations.log','ERROR')
            try:  
                model_name_xgboost= f'XGBOOST_Regressor_cluster_no_{cluster_no}.sav'
                path_xgboost=os.path.join(os.getcwd(),'models',model_name_xgboost)
                pickle.dump(self.rn, open(path_xgboost, 'wb'))
                self.log.log_writer(f'Successfully saved the {model_name_xgboost} at path {path_xgboost}','model_operations.log')
            except Exception as e:
                try:
                    self.log.log_writer(f'Could not saved the {model_name_xgboost} at path {path_xgboost} error: {str(e)}','model_operations.log','ERROR')
                except Exception as NameError:
                    self.log.log_writer(f'NameError occured in train_model_with_clusters','model_operations.log','ERROR')
    def predict_model_with_cluster(self,x_test,y_test):
        y_true_random_forest=[]
        y_pred_random_forest=[]
        y_true_random_forest_XGBOOST_Regressor=[]
        y_pred_random_forest_XGBOOST_Regressor=[]
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
                            y_true_random_forest=y_true_random_forest+cluster_y.to_numpy().tolist()
                            y_pred_random_forest=y_pred_random_forest+predict_random_forest.tolist()
                            random_forest_file=file
                            self.log.log_writer(f'Sucessfully load predict the model {random_forest_file}','model_operations.log')
                        except Exception as e:
                            self.log.log_writer(f'Couldnot load predict the model {random_forest_file} error {str(e)}','model_operations.log','Error')

                    elif 'XGBOOST_Regressor' in file:
                        try:
                            model_XGBOOST_Regressor=pickle.load(open(os.path.join(path,file),'rb'))
                            predict_model_XGBOOST_Regressor=model_XGBOOST_Regressor.predict(cluster_x.drop(['cluster'],axis=1).to_numpy())
                            y_true_random_forest_XGBOOST_Regressor=y_true_random_forest_XGBOOST_Regressor+cluster_y.to_numpy().tolist()
                            y_pred_random_forest_XGBOOST_Regressor=y_pred_random_forest_XGBOOST_Regressor+predict_model_XGBOOST_Regressor.tolist()
                            XGBOOST_Regressor_file=file
                            self.log.log_writer(f'Sucessfully load predict the model {XGBOOST_Regressor_file}','model_operations.log')
                        except Exception as e:
                            self.log.log_writer(f'Couldnot load predict the model {XGBOOST_Regressor_file} error {str(e)}','model_operations.log','Error')
            random_forest_r2_score=self.metrics_obj.compute_r2_score(y_true=cluster_y,y_pred=predict_random_forest)
            XGBOOST_Regressor_metrics_r2_score=self.metrics_obj.compute_r2_score(y_true=cluster_y,y_pred=predict_model_XGBOOST_Regressor)
            self.log.log_writer(f'The r2_score for random_forest is {random_forest_r2_score} and xgboost regressor is {XGBOOST_Regressor_metrics_r2_score} for cluster no {cluster_no}','model_operations.log')

            if XGBOOST_Regressor_metrics_r2_score>random_forest_r2_score:
                try:
                    print("@@")
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

        return y_pred_random_forest,y_pred_random_forest,y_true_random_forest_XGBOOST_Regressor,y_pred_random_forest_XGBOOST_Regressor
        