from src.Training_Service.Data_Validation import Data_Validation
from src.Training_Service.db import db_operations
from flask import Flask, render_template, jsonify,request,redirect,url_for
import os
import shutil
import sys
app=Flask(__name__)
def createTrainingDirectories():
    if 'Training_Batch_Files' in os.listdir(os.getcwd()):
        shutil.rmtree('Training_Batch_Files')
    os.makedirs('Training_Batch_Files')
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')
@app.route('/train',methods=['POST','GET'])
def train():
    if request.method=='POST':
        createTrainingDirectories()
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
        return '<h1>Cool! Training Completed Sucessfully!</h1>'
    else:
        return redirect(url_for('home'))
        return app.route('/')
if __name__=='__main__':
    app.run()
