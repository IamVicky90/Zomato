from src.Training_Service.Data_Validation import Data_Validation
from src.Training_Service.db import db_operations
from src.Data_Preprocessing import preprocessing
from src.Cluster_Data import cluster
from src.Training_Service.model_operations import model_ops
from flask import Flask, render_template, request,redirect,url_for
from src.Prediction_Service.Data_Validation import Prediction_Data_Validation
from src.Prediction_Service.db import db_operations_prediction
from src.Prediction_Service.predict_data import prediction
import os
import shutil
import sys
import yaml
import ast
app=Flask(__name__)
def create_Traing_necessary_Directories():
    if 'Training_Batch_Files' in os.listdir(os.getcwd()):
        shutil.rmtree('Training_Batch_Files')
    os.makedirs('Training_Batch_Files')
    if 'Bad_Data_Folder' in os.listdir(os.getcwd()):
        shutil.rmtree('Bad_Data_Folder')
    os.makedirs('Bad_Data_Folder')
    if 'Good_Data_Folder' in os.listdir(os.getcwd()):
        shutil.rmtree('Good_Data_Folder')
    os.makedirs('Good_Data_Folder')
    if 'Master_Training_File' in os.listdir(os.getcwd()):
        shutil.rmtree('Master_Training_File')
    if 'models' in os.listdir(os.getcwd()):
        shutil.rmtree('models')
    os.makedirs('models')
def create_Prediction_necessary_Directories():
    if 'Prediction_Batch_Files' in os.listdir(os.getcwd()):
        shutil.rmtree('Prediction_Batch_Files')
    os.makedirs('Prediction_Batch_Files')
    if 'Prediction_Bad_Data_Folder' in os.listdir(os.getcwd()):
        shutil.rmtree('Prediction_Bad_Data_Folder')
    os.makedirs('Prediction_Bad_Data_Folder')
    if 'Prediction_Good_Data_Folder' in os.listdir(os.getcwd()):
        shutil.rmtree('Prediction_Good_Data_Folder')
    os.makedirs('Prediction_Good_Data_Folder')
    if 'Master_Prediction_File' in os.listdir(os.getcwd()):
        shutil.rmtree('Master_Prediction_File')
    if 'Prediction File' in os.listdir(os.getcwd()):
            shutil.rmtree('Prediction File')
    os.makedirs('Prediction File')
def read_params():
    with open('params.yaml') as  file:
        config=yaml.safe_load(file)
    return config
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')
@app.route('/train',methods=['POST','GET'])
def train():
    if request.method=='POST':
        create_Traing_necessary_Directories()
        Training_path='Training_Batch_Files'
        files=request.files.getlist("folder")
        for file in files:
            if '.csv' in file.filename:
                basepath = os.getcwd()
                if sys.platform=='linux':
                    file_name=file.filename.split('/')[-1]
                elif sys.platform=='win32':
                    file_name=file.filename.split('\\')[-1]
                else:
                    file_name=file.filename.split('/')[-1]
                file_path = os.path.join(
                basepath, Training_path,file_name)
                file.save(file_path)
            else:
                return '<h1>OOPS! There are other files too! </br> Please Enter only CSV Files Folder</h1>'
        try:
            dv=Data_Validation.Validate_data('Training_Batch_Files','schema_training.json')
            dv.validate_name_of_the_files()
            dv.validate_number_of_columns()
            dv.validate_name_of_columns()
            dv.replace_Null_with_NAN()
            db=db_operations.db_ops()
            db.create_Table('Zomato','zomato.db')
            db.insert_values_into_table('zomato.db','Zomato')
            db.dump_data_from_database_to_one_csv_file('zomato.db','Zomato')
            process=preprocessing.process_data()
            config_file=read_params()
            df=process.create_csv_to_dataframe(config_file['data_preprocessing']['csv_path'])
            df=process.drop_unnecessery_columns(df,config_file['data_preprocessing']['columns_to_remove'])
            df=process.rename_the_columns(df,config_file['data_preprocessing']['rename_columns'])
            # df=process.impute_nan_values_by_knn_imputer(df,config_file['data_preprocessing']['n_neighbors'])
            df=process.drop_nan_values(df)
            df= process.cleaning_the_data_present_in_the_features(df)
            dummy= process.create_dummy_columns(df,columns=config_file['data_preprocessing']['columns_to_dummy_variables'],drop_first=config_file['data_preprocessing']['drop_first'])
            X_dummy,Y=process.split_dummy_into_X_and_Y(dummy)
            X_dummy_selected_list_of_features_by_lasso=process.feature_selection(X_dummy,Y,alpha=config_file['data_preprocessing']['alpha'])
            final_x_train=process.return_selected_features_by_lasso(X_dummy,X_dummy_selected_list_of_features_by_lasso)
            x_train,x_test,y_train,y_test=process.split_into_train_test(final_x_train,Y,test_size=config_file['data_preprocessing']['train_test_split']['test_size'],random_state=config_file['data_preprocessing']['train_test_split']['random_state'])
            cluster_obj=cluster.cluster()
            x_train_with_cluster=cluster_obj.create_clusters(x_train)
            x_test_with_cluster_column=cluster_obj.predict_clusters(x_test)
            model_obs_obj=model_ops.model_operations()
            parameters_random_forest_grid=config_file['hyperparameter_tunning']['random_forest_hyperparameters']['parameters']
            cv_random_forest_grid=config_file['hyperparameter_tunning']['random_forest_hyperparameters']['GridSearchCV']['cv']
            verbose_random_forest_grid=config_file['hyperparameter_tunning']['random_forest_hyperparameters']['GridSearchCV']['verbose']
            parameters_random_forest_random=config_file['hyperparameter_tunning']['random_forest_hyperparameters']['parameters']
            cv_random_forest_random=config_file['hyperparameter_tunning']['random_forest_hyperparameters']['RandomizedSearchCV']['cv']
            verbose_random_forest_random=config_file['hyperparameter_tunning']['random_forest_hyperparameters']['RandomizedSearchCV']['verbose']
            parameters_xgboost_grid=config_file['hyperparameter_tunning']['xgboost_hyperparameters']['parameters']
            cv_xgboost_grid=config_file['hyperparameter_tunning']['xgboost_hyperparameters']['GridSearchCV']['cv']
            verbose_xgboost_grid=config_file['hyperparameter_tunning']['xgboost_hyperparameters']['GridSearchCV']['verbose']
            parameters_xgboost_random=config_file['hyperparameter_tunning']['xgboost_hyperparameters']['parameters']
            cv_xgboost_random=config_file['hyperparameter_tunning']['xgboost_hyperparameters']['RandomizedSearchCV']['cv']
            verbose_xgboost_random=config_file['hyperparameter_tunning']['xgboost_hyperparameters']['RandomizedSearchCV']['verbose']
            model_obs_obj.train_model_with_clusters_with_hyperparameter_tuning(x_train_with_cluster,y_train,x_test_with_cluster_column,y_test,parameters_random_forest_grid,cv_random_forest_grid,verbose_random_forest_grid,parameters_random_forest_random,cv_random_forest_random,verbose_random_forest_random,parameters_xgboost_grid,cv_xgboost_grid,verbose_xgboost_grid,parameters_xgboost_random,cv_xgboost_random,verbose_xgboost_random)
            model_obs_obj.selct_best_model_with_cluster(x_test_with_cluster_column,y_test)
            
            return '<h1>Cool! Training Completed Sucessfully!</h1>'
        except Exception as e:
            return f'<h1>We are facing some error: {str(e)} <h1>'
    else:
        return redirect(url_for('home'))
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        create_Prediction_necessary_Directories()
        prediction_path='Prediction_Batch_Files'
        files=request.files.getlist("folder")
        for file in files:
            if '.csv' in file.filename:
                basepath = os.getcwd()
                if sys.platform=='linux':
                    file_name=file.filename.split('/')[-1]
                elif sys.platform=='win32':
                    file_name=file.filename.split('\\')[-1]
                else:
                    file_name=file.filename.split('/')[-1]
                file_path = os.path.join(
                basepath, prediction_path,file_name)
                file.save(file_path)
            else:
                return '<h1>OOPS! There are other files too! </br> Please Enter only CSV Files Folder</h1>'
        try:
            dv=Prediction_Data_Validation.Validate_data(prediction_path,'schema_prediction.json')
            dv.validate_name_of_the_files()
            dv.validate_number_of_columns()
            dv.validate_name_of_columns()
            dv.replace_Null_with_NAN()
            db=db_operations_prediction.db_ops()
            db.create_Table('Zomato','zomato_prediction.db')
            db.insert_values_into_table('zomato_prediction.db','Zomato')
            db.dump_data_from_database_to_one_csv_file('zomato_prediction.db','Zomato')
            process=preprocessing.process_data()
            config_file=read_params()
            df=process.create_csv_to_dataframe(config_file['data_preprocessing']['predict_csv_path'],service_name='p')
            df=process.drop_unnecessery_columns(df,config_file['data_preprocessing']['columns_to_remove'],service_name='p')
            df=process.rename_the_columns(df,config_file['data_preprocessing']['rename_columns'],service_name='p')
            # df=process.impute_nan_values_by_knn_imputer(df,config_file['data_preprocessing']['n_neighbors'])
            df=process.drop_nan_values(df,service_name='p')
            df= process.cleaning_the_data_present_in_the_features(df,service_name='p')
            dummy= process.create_dummy_columns(df,columns=config_file['data_preprocessing']['columns_to_dummy_variables'],drop_first=config_file['data_preprocessing']['drop_first'],service_name='p')
            with open('src/log_files/Training_logs/X_dummy_selected_list_of_features_by_lasso.txt', 'r') as file:
                features_that_are_taken_by_feature_selection=ast.literal_eval(file.read())
            final_x_train=process.return_selected_features_by_lasso(dummy,features_that_are_taken_by_feature_selection)

            cluster_obj=cluster.cluster()
            x_test_with_cluster_column=cluster_obj.predict_clusters(final_x_train,service_name='p')
            prediction_obj=prediction.model_prediction()
            total_predictions_from_model=prediction_obj.model_prediction_with_cluster(x_test_with_cluster_column)
            return f'<h1>Cool! Prediction Completed Sucessfully here are the predictions results!</h1> </br> {total_predictions_from_model} and csv file is saved at path Prediction File/prediction.csv'
        except Exception as e:
            return f'<h1>We are facing some error: {str(e)} <h1>'
    else:
        return redirect(url_for('home'))
if __name__=='__main__':
    app.run(debug=True)
