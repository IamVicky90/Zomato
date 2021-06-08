import os
import datetime
class log:
    def __init__(self) -> None:
        pass
    def log_writer(self,message,filename,message_type='INFO',service_name='t'):
        '''
        Params desc:
            message: Write your message for logger file,
            filename: Name of the file,
            message_type: Type of the Message,
            service_name: Enter the service_name to save the files in that service folder:
                Expected params: 't' for training, 'p' for the prediction service
        '''
        if service_name=='t':
            service_path='Training_logs'
        elif service_name=='p':
            service_path='Prediction_logs'
        else:
            raise Exception(f'Unknown service name {service_name}, exppected arguments, t and p.')
        os.makedirs(os.path.join('src','log_files'),exist_ok=True)
        full_service_path=os.path.join('src','log_files',service_path)        
        os.makedirs(full_service_path,exist_ok=True)
        self.now = datetime.datetime.now()
        with open(os.path.join(full_service_path,filename),'a+') as f:
            f.write(str(self.now.strftime("%Y-%m-%d %H:%M:%S"))+"\t\t"+message_type+": "+str(message)+"\n")       
            f.close() 