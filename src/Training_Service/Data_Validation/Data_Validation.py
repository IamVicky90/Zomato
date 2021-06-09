from src.logger import logger
import os
import json
import re
import shutil
class Validate_data:
    def __init__(self,data_folder,schema):
        self.data_folder=data_folder
        self.schema=schema
        self.log=logger.log()
        self.json_file=json.load(open(self.schema,))
    def validate_name_of_the_files(self):
        for file in os.listdir(self.data_folder):
            if re.match('[zomato_]+[\d]+[_]+[\d]+[.csv]',file) and '.dvc' not in file:
                self.log.log_writer(f'Regex Matches for file {file}','Data_Validation.log')
                name,_=file.split('.')
                _,LengthOfDateStampInFile, LengthOfTimeStampInFile=name.split('_')
                if len(LengthOfDateStampInFile)==int(self.json_file['LengthOfDateStampInFile']) and len(LengthOfTimeStampInFile)==int(self.json_file['LengthOfTimeStampInFile']):
                    self.log.log_writer(f'LengthOfDateStampInFile and LengthOfTimeStampInFile,  Matches for file {file}','Data_Validation.log')
                    self.move_Good_Data_Folder(os.path.join('Training_Batch_Files',file),file)
                else:
                    self.log.log_writer(f'LengthOfDateStampInFile and LengthOfTimeStampInFile,  not Matches for file {file}','Data_Validation.log',message_type='Warning')
            else:
                self.log.log_writer(f'Regex not matches for file {file}','Data_Validation.log','Warning')

        return
    def move_Good_Data_Folder(self,file_path,file):
        os.makedirs('Good_Data_Folder',exist_ok=True)
        try:
            shutil.move(file_path,'Good_Data_Folder')
            self.log.log_writer(f'Sucessfully copy the file {file} to Good_Data_Folder','Data_Validation.log')
        except Exception as e:
            self.log.log_writer(f'Could not copy the file {file} to Good_Data_Folder error: {str(e)}','Data_Validation.log','Error')
    def copy(self):
        pass
