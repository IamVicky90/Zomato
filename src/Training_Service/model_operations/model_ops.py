from src.logger import logger
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
class model_operations:
    def __init__(self):
        self.log=logger.log()
        self.rn=RandomForestRegressor()
    def train_model_with_clusters(self,x,y):
        self.log.log_writer('Model Operations started!','model_operations.log')
        try:
            for cluster_no in x['cluster'].unique():
                cluster_x=x[x['cluster']==cluster_no]
                cluster_y=y[cluster_x.index]
                self.rn.fit(cluster_x.drop(['cluster'],axis=1),cluster_y)
                model_name= f'random_forest_regressor_cluster_no_{cluster_no}.sav'
                path=os.path.join(os.getcwd(),'models',model_name)
                pickle.dump(self.rn, open(path, 'wb'))
                self.log.log_writer(f'Successfully saved the {model_name} at path {path}','model_operations.log')
        except Exception as e:
            try:
                self.log.log_writer(f'Could not saved the {model_name} at path {path} error: {str(e)}','model_operations.log','ERROR')
            except Exception as NameError:
                self.log.log_writer(f'NameError occured in train_model_with_clusters','model_operations.log','ERROR')
    def predict_model_with_cluster(self,x_test,y_test):
        for cluster_no in x_test['cluster'].unique():
                cluster_x=x_test[x_test['cluster']==cluster_no]
                cluster_y=y_test[cluster_x.index]
                path=os.path.join(os.getcwd(),'models')
                for file in path:
                    if cluster_no in file and '.sav' in file:
                        model=pickle.load(open(os.path.join(path,file),'rb'))
                        p=model.predict(cluster_x)
                        print(p)

                # self.rn.fit(cluster_x.drop(['cluster'],axis=1),cluster_y)


        
