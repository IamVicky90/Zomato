from src.Training_Service.Data_Validation import Data_Validation

obj=Data_Validation.Validate_data('Training_Batch_Files','schema_training.json')
obj.validate_name_of_the_files()
