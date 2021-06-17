from src.Training_Service.Data_Validation import Data_Validation
from src.Training_Service.db import db_operations
from src.Training_Service.Data_Preprocessing import preprocessing
from src.Training_Service.Cluster_Data import cluster
from src.Training_Service.model_operations import model_ops
from flask import Flask, render_template, jsonify,request,redirect,url_for
import os
import shutil
import sys
import yaml
app=Flask(__name__)
def create_necessary_Directories():
    if 'Training_Batch_Files' in os.listdir(os.getcwd()):
        shutil.rmtree('Training_Batch_Files')
    os.makedirs('Training_Batch_Files')
    if 'models' in os.listdir(os.getcwd()):
        shutil.rmtree('models')
    os.makedirs('models')
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
        create_necessary_Directories()
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
        model_obs_obj=model_ops.model_operations()
        model_obs_obj.train_model_with_clusters(x_train_with_cluster,y_train)
        x_test_with_cluster_column=cluster_obj.predict_clusters(x_test)
        y_pred_random_forest,y_pred_random_forest,y_true_random_forest_XGBOOST_Regressor,y_pred_random_forest_XGBOOST_Regressor=model_obs_obj.predict_model_with_cluster(x_test_with_cluster_column,y_test)
        return '<h1>Cool! Training Completed Sucessfully!</h1>'
    else:
        return redirect(url_for('home'))
if __name__=='__main__':
    app.run(debug=True)
