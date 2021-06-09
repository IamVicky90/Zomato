from src.logger import logger
import os
import json
import re
import shutil
import csv
class Validate_data:
    def __init__(self,data_folder,schema):
        self.data_folder=data_folder
        self.schema=schema
        self.log=logger.log()
        self.json_file=json.load(open(self.schema,))
        os.makedirs('Good_Data_Folder',exist_ok=True)
        self.good_data_path=os.path.join(os.getcwd(),'Good_Data_Folder')
        os.makedirs('Bad_Data_Folder',exist_ok=True)
        self.bad_data_path=os.path.join(os.getcwd(),'Bad_Data_Folder')
    def validate_name_of_the_files(self):
        for file in os.listdir(self.data_folder):
            if re.match('[zomato_]+[\d]+[_]+[\d]+[.csv]',file) and '.dvc' not in file:
                self.log.log_writer(f'Regex Matches for file {file}','Data_Validation.log')
                name,_=file.split('.')
                _,LengthOfDateStampInFile, LengthOfTimeStampInFile=name.split('_')
                if len(LengthOfDateStampInFile)==int(self.json_file['LengthOfDateStampInFile']) and len(LengthOfTimeStampInFile)==int(self.json_file['LengthOfTimeStampInFile']):
                    self.log.log_writer(f'LengthOfDateStampInFile and LengthOfTimeStampInFile,  Matches for file {file} and copy to Good Data Folder','Data_Validation.log')
                    self.copy_to_Good_Data_Folder(os.path.join('Training_Batch_Files',file),file)
                else:
                    self.log.log_writer(f'LengthOfDateStampInFile and LengthOfTimeStampInFile,  not Matches for file {file} and Copy to Bad Data Folder','Data_Validation.log',message_type='Warning')
                    self.copy_to_Bad_Data_Folder(os.path.join('Training_Batch_Files',file),file)
            else:
                self.log.log_writer(f'Regex not matches for file {file} and copy to Bad Data Folder','Data_Validation.log','Warning')
                self.copy_to_Bad_Data_Folder(os.path.join('Training_Batch_Files',file),file)

        return
    def copy_to_Good_Data_Folder(self,file_path,file):
        try:
            shutil.copy(file_path,'Good_Data_Folder')
            self.log.log_writer(f'Sucessfully copy the file {file} to Good_Data_Folder','Data_Validation.log')
        except Exception as e:
            self.log.log_writer(f'Could not copy the file {file} to Good_Data_Folder error: {str(e)}','Data_Validation.log','Error')
    def copy_to_Bad_Data_Folder(self,file_path,file):
        try:
            shutil.copy(file_path,'Bad_Data_Folder')
            self.log.log_writer(f'Sucessfully copy the file {file} to Bad_Data_Folder','Data_Validation.log')
        except Exception as e:
            self.log.log_writer(f'Could not copy the file {file} to Bad_Data_Folder error: {str(e)}','Data_Validation.log','Error')
    def move_to_Bad_Data_Folder(self,file_path,file):
        try:
            shutil.move(file_path,'Bad_Data_Folder')
            self.log.log_writer(f'Sucessfully copy the file {file} to Bad_Data_Folder','Data_Validation.log')
        except Exception as e:
            self.log.log_writer(f'Could not copy the file {file} to Bad_Data_Folder error: {str(e)}','Data_Validation.log','Error')
    def validate_number_of_columns(self):
        for file in os.listdir(self.good_data_path):
            with open(file,'') as csvfile:
                reader=csv.reader(csvfile)
                for row in reader:
                    if len(row)==int(self.json_file['NumberofColumns']):
                        self.log.log_writer(f'Number of Columns of file {file} Matched with the schema','Data_Validation.log')
                    else:
                        self.log.log_writer(f'Number of Columns of file  {file} doesnot Matched with the schema so we are moving it to Bad Data','Data_Validation.log','Warning')
                        self.move_to_Bad_Data_Folder()
                    break

   
