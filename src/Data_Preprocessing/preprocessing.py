from src.logger import logger
import pandas as pd 
from sklearn.impute import KNNImputer # To fill null values
import numpy as np # For array manipulation
from sklearn.feature_selection import SelectFromModel # For feature Selection
from sklearn.linear_model import Lasso # Import Lasso as feature Selection model
from sklearn.model_selection import train_test_split
class process_data:
    def __init__(self):
        self.log=logger.log()
    def create_csv_to_dataframe(self,path,service_name='t'):
        try:
            df= pd.read_csv(path)
            self.log.log_writer(f'Successfully converted csv file from path {path} to Pandas DataFrame','preprocessing.log', service_name=service_name)
        except Exception as e:
            self.log.log_writer(f'Could not convert csv file from path {path} to Pandas DataFrame error: {str(e)}','preprocessing.log',message_type='Error')
        return df
    def drop_unnecessery_columns(self,df,columns_to_remove,service_name='t'):
        try:
            df=self.create_csv_to_dataframe('Master_Training_File/Zomato.csv')
            df.drop(columns_to_remove,axis=1,inplace=True)
            self.log.log_writer(f'Successfully removed the columns {columns_to_remove} from Pandas DataFrame','preprocessing.log',service_name=service_name)
        except Exception as e:
            self.log.log_writer(f'Couldnnot removed the columns {columns_to_remove} from Pandas DataFrame error: {str(e)}','preprocessing.log',service_name=service_name)
        return df
    def rename_the_columns(self,df,dictionary,service_name='t'):
        try:
            df.rename(columns={'listed_in_type':'type','listed_in_city':'city','approx_cost_for_two_people':'cost'},inplace=True) #Rename the column for the simplicity
            self.log.log_writer(f'Successfully renames the columns from {dictionary.keys()} to {dictionary.values()} in Pandas DataFrame','preprocessing.log',service_name=service_name)
        except Exception as e:
            self.log.log_writer(f'Could not renames the columns from {dictionary.keys()} to {dictionary.values()} in Pandas DataFrame error: {str(e)}','preprocessing.log','Error',service_name=service_name)
        return df
    def impute_nan_values_by_knn_imputer(self,df,n_neighbors):
        # Due to a lot of catagorical features we doesn't use this in this scenerio
        null_values=df.isnull().sum().sum()
        if null_values>0:
            try:
                knn=KNNImputer(n_neighbors=n_neighbors)
                numpy_array_data=knn.fit_transform(df.drop(['cost'],axis=1),df['cost'])
                df=pd.DataFrame(numpy_array_data,columns=df.drop(['cost'],axis=1).columns)
                self.log.log_writer(f'There is {null_values} nan values present in this dataset that are sucessfully handeled by KNNImputer','preprocessing.log',message_type='Warning')
            except Exception as e:
                self.log.log_writer(f'There is {null_values} nan values present in this dataset that are not handeled by KNNImputer due to an error: {str(e)}','preprocessing.log',message_type='Error')

        else:
            self.log.log_writer('There is no nan values in this dataset','preprocessing.log',message_type='Warning')
        return df
    def drop_nan_values(self,df,service_name='t'):
        null_values=df.isnull().sum().sum()
        if null_values>0:
            try:
                df.dropna(inplace=True)
                self.log.log_writer(f'There is {null_values} nan values present in this dataset that are sucessfully removed','preprocessing.log',message_type='Warning',service_name=service_name)
            except Exception as e:
                self.log.log_writer(f'There is {null_values} nan values present in this dataset that are not removed due to an error: {str(e)}','preprocessing.log',message_type='Error',service_name=service_name)
        else:
            self.log.log_writer('There is no nan values in this dataset','preprocessing.log',message_type='Warning',service_name=service_name)
        return df
    def cleaning_the_data_present_in_the_features(self,df,service_name='t'):
        try:
            df['rate']=df['rate'].str.replace('/5','')#Rplace '/5' with '' to convert it into number(float in this case)
            df=df.loc[df.rate!='NEW'] #DataFrame without NEW values in rate colum
            df['rate']=df['rate'].astype('float64') # convert the datatype to float64
            self.log.log_writer(f'Sucessfuly cleaned and change the datatype of feature rate','preprocessing.log',message_type='INFO',service_name=service_name)
        except Exception as e:
            self.log.log_writer(f'Could not cleaned and change the datatype of feature rate due to error: {str(e)}','preprocessing.log',message_type='Error',service_name=service_name)
        try:
            remove_comma= lambda x:x.replace(',','') if type(x)==np.str else x #create a lambda function to remove ',' with empty string ''
            df.cost=df.cost.apply(remove_comma) #apply the funtion remove_comma to extract comma out of it.
            df.cost=df.cost.astype('int64') # convert the datatype to int64
            self.log.log_writer(f'Sucessfuly cleaned and change the datatype of feature cost','preprocessing.log',message_type='INFO',service_name=service_name)
        except Exception as e:
            self.log.log_writer(f'Could not cleaned and change the datatype of feature cost due to error {str(e)}','preprocessing.log',message_type='Error',service_name=service_name)
        return df
    def create_dummy_columns(self,df,columns,drop_first=True,service_name='p'):
        try:
            dummy=pd.get_dummies(df,columns=columns,drop_first=drop_first)
            self.log.log_writer(f'Sucessfully created the dummy variables of columns {columns} now the column size is increased from {df.shape[1]} to {dummy.shape[1]}','preprocessing.log',message_type='INFO',service_name=service_name)
        except Exception as e:
            self.log.log_writer(f'Could not create the dummy variables of columns {columns} due to error: {str(e)}','preprocessing.log',message_type='Error',service_name=service_name)
        return dummy
    def split_dummy_into_X_and_Y(self,dummy):
        try:
            X_dummy=dummy.drop(['cost'],axis=1) # Seperate the independent features
            Y_dummy=dummy['cost'] # Seperate the dependent ('cost') from independent feature
            self.log.log_writer(f'Sucessfully split the dummy dataframe into X_dummy, Y_dummy','preprocessing.log',message_type='INFO')
        except Exception as e:
            self.log.log_writer(f'Could not split the dummy dataframe into X_dummy,Y_dummy error: {str(e)}','preprocessing.log',message_type='Error')
        return X_dummy,Y_dummy
    def feature_selection(self,X_dummy,Y_dummy,alpha=0.005):
        '''This will returned the features that are important this is helpful to increase the accuracy and decrease the computational cost '''
        try:
            sel_features=SelectFromModel(Lasso(alpha=alpha))  # Initialize SelectFromModel along Lasso with 0.005 alpha values
            sel_features.fit(X_dummy,Y_dummy) # Fiting the model
            X_dummy_selected_list_of_features_by_lasso=list(X_dummy.columns[sel_features.get_support()]) # returns the colums choose by Lasso
            self.log.log_writer(f'Sucessfully created the list of the features that are important for the prediction','preprocessing.log',message_type='INFO')
            self.log.log_writer(f'To see the list of the features that are important for the prediction please see the X_dummy_selected_list_of_features_by_lasso.txt file','preprocessing.log',message_type='Warning')
            # self.log.log_writer(str(X_dummy_selected_list_of_features_by_lasso),'X_dummy_selected_list_of_features_by_lasso.txt',message_type='INFO')
            with open('src/log_files/Training_logs/X_dummy_selected_list_of_features_by_lasso.txt','w') as txt:
                txt.write(str(X_dummy_selected_list_of_features_by_lasso))
                txt.close()
        except Exception as e:
            self.log.log_writer(f'Could not create the list of the features that are important for the prediction error: {str(e)}','preprocessing.log',message_type='Error')
        return X_dummy_selected_list_of_features_by_lasso
    def return_selected_features_by_lasso(self,x_dummy,X_dummy_selected_list_of_features_by_lasso):
        try:
            dummy_dataframe_with_selected_features_by_lasso=x_dummy[X_dummy_selected_list_of_features_by_lasso]
            self.log.log_writer(f'Sucessfully create the dataframe of the selected features that are important for the prediction.','preprocessing.log',message_type='INFO')
        except Exception as e:
            self.log.log_writer(f'Could not create the dataframe of the selected features that are important for the prediction error: {str(e)}','preprocessing.log',message_type='Error')
        return dummy_dataframe_with_selected_features_by_lasso

    def split_into_train_test(self,final_x_train,Y,test_size=0.2,random_state=42):
        try:
            x_train,x_test,y_train,y_test=train_test_split(final_x_train,Y,test_size=0.2,random_state=42) #split into x_train,x_test,y_train,y_test with 20% of test size
            self.log.log_writer(f'Sucessfully train_test_split the dummy dataframe into x_train,x_test,y_train,y_test','preprocessing.log',message_type='INFO')
        except Exception as e:
            self.log.log_writer(f'Could not train_test_split the dummy dataframe into x_train,x_test,y_train,y_test error: {str(e)}','preprocessing.log',message_type='Error')
        return x_train,x_test,y_train,y_test
    