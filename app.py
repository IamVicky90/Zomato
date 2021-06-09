from src.Training_Service.Data_Validation import Data_Validation
from src.Training_Service.db import db_operations
dv=Data_Validation.Validate_data('Training_Batch_Files','schema_training.json')
dv.validate_name_of_the_files()
dv.validate_number_of_columns()
dv.validate_name_of_columns()
dv.replace_Null_with_NAN()
db=db_operations.db_ops()
db.create_Table('Zomato','zomato.db')
db.insert_values_into_table('zomato.db','Zomato')
db.dump_data_from_database_to_one_csv_file('zomato.db','Zomato')
