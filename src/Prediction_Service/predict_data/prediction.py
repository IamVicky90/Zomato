from src.logger import logger
import os
import pickle
import pandas as pd
class model_prediction:
    def __init__(self):
        self.log=logger.log()
    def model_prediction_with_cluster(self,x_data):
        self.log.log_writer('Model Prediction Part Starting!','prediction.log',service_name='p')
        path=os.path.join(os.getcwd(),'models')
        total_predictions_from_model=[]
        total_cluster_x_data_id=[]
        for cluster_no in x_data['cluster'].unique():
            cluster_x=x_data[x_data['cluster']==cluster_no]
            for file in os.listdir(path):
                if str(cluster_no) in file and '.sav' in file:
                    try:
                        model=pickle.load(open(os.path.join(path,file),'rb'))
                        self.log.log_writer(f'Sucessfully load the model {file} for cluster {cluster_no}','prediction.log',service_name='p')
                    except Exception as e:
                        self.log.log_writer(f'Could not load the model {file} for cluster {cluster_no} error: {str(e)}','prediction.log',message_type='Error',service_name='p')
                    prediction_from_model=model.predict(cluster_x.drop(['cluster'],axis=1).to_numpy())
                    try:
                        self.log.log_writer(f'Sucessfully predict the model {file} for cluster {cluster_no} ','prediction.log',service_name='p')
                    except Exception as e:
                        self.log.log_writer(f'Could not predict the model {file} for cluster {cluster_no} error: {str(e)}','prediction.log',message_type='Error',service_name='p')
                    total_predictions_from_model=total_predictions_from_model+list(prediction_from_model)
                    total_cluster_x_data_id=total_cluster_x_data_id+list(cluster_x.index.to_numpy())
        try:
            df=pd.DataFrame(total_predictions_from_model,columns=['Prediction'])
            df['id']=total_cluster_x_data_id
            df.set_index('id',inplace=True)
            df.sort_index(inplace=True)
            df.to_csv(os.path.join('Prediction File','prediction.csv'),index=False)
            self.log.log_writer(f'Sucessfully convert the prediction and id array into pandas Data Frame and successfully return the csv Prediction file to path, File/prediction.csv','prediction.log',service_name='p')
            return total_predictions_from_model
        except Exception as e:
            self.log.log_writer(f'Could not convert the prediction and id array into pandas Data Frame and could not return the csv Prediction file to path, File/prediction.csv','prediction.log',service_name='p')