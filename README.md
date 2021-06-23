## Table of Contents.
  * [Demo](#demo)
  * Deployment
  * [Data Validation](#data-validation)
  * [Data Base Operations](#data-base-operations)
  * [Data Preprocessing](#data-preprocessing)
  * [Clustering the data](#clustering-the-data)
  * [Hyperparameter Tunning](#hyperparameter-tunning)
  * [Data Prediction](#data-prediction)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Directory Tree](#directory-tree)
  * [Libraries](#libraries)
# Demo
![error check your internet](https://github.com/IamVicky90/Zomato/blob/dev/src/demo.PNG)
### [Deployment is done in heroku with ci-cd pipeline](https://zomato-price-prediction.herokuapp.com/)
# Steps That I take to Solve this Machine Learning Problem
## Data Validation
------------
1. In this step, we perform different sets of validation on the given set of training files.
2. Name Validation - We validate the name of the files based on the given name in the schema file. We have created a regex pattern as per the name given in the schema file to use for validation. After validating the pattern in the name, we check for the length of date in the file name as well as the length of time in the file name. If all the values are as per requirement, we copy such files to "Good_Data_Folder" else we copy such files to "Bad_Data_Folder."
3. Number of Columns - We validate the number of columns present in the files, and if it doesn't match with the value given in the schema file, then the file is copied to "Bad_Data_Folder."
4. Name of Columns - The name of the columns is validated and should be the same as given in the schema file. If not, then the file is copied to "Bad_Data_Folder".
5. Null values in columns - If any of the columns in a file have all the values as NULL or missing, we discard such a file and move it to "Bad_Data_Folder".
6. Replace Null with NAN - Replace the null values with 'nan' string because in some databases we can't add Null values or Null type values.
7. The datatype of columns - The datatype of columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is copied to "Bad_Data_Folder".
------------
## Data Base Operations
1. In this stage we have to dump our data into some databases so that we or the team members can acess it.
2. The database that we have use in this problem is sqllite3 for our simplicity.
3. We also create a csv master file out of this sqllite database so that we can do all the operation without damaging the dataset
------------
## Data Preprocessing
1. In this stage we will do the data preprocessing that includes cleaning the data/columns, EDA(Exploratory Data Analysis),feature engineering, removing or imputing nan values, feature selection and splitting the data into train_test_split.
### Steps that included in Data Preprocessing
1. cleaning the data/columns:
    Removed '/5' or 'new' from 'rate' and converts it datatype fron str to 'float64'. likewise we also remove comma from 'cost' and change it's data type from str to 'int64'.
2. I removed all the NAN values, I also make a funtion of knnImputer but due to 90 percent of overall the values are catagorical so it is not a good idea to impute these values with this so I comment it.

2. Feature engineering:
    1. Drop unnecessery columns:
        In 'Drop unnecessery columns' stage we remove 'serial', 'url', 'address' , 'phone', 'reviews_list', 'dish_liked' columns as they don't play any role while predicting or analyzing the data.
    2. Rename the columns:
        Rename the columns from 'listed_in_type', 'listed_in_city', 'approx_cost_for_two_people' to 'type', 'city', 'cost' for our simplicity.
    3. First I used ordinal encoding but it doesn't gives me a better accuracy and in fact it is not a right or appropriate way to use when the data in catagorical feature don't have any relation between them.
    4. I used onehot encoding and category_encoders with base value 5( we use this technique when we face a lot of different catagarical data, so that it doesn't lead to curse of dimenstionality) because the data in catagorical feature don't have any relation between them.
    5. After applying onehot encoding and I see that onehot encoding gives us the better result so I keep onehot encoding.
3. Exploratory Data Analysis:
    1. All the Exploratory Data Analysis is done in [Zomata EDA.ipynb](https://github.com/IamVicky90/Zomato/blob/dev/notebooks/Zomata%20EDA.ipynb) along with the proper comment.
    2. It is recommended to see this jupyter notebook file, I have explained all the steps beautifully here.
4. Feature Selection:
    1. As there is a lot of features present so that may cause a curse of dimenstionality, to remove this I selected features using Lasso by SelectFromModel funtion from sklearn library.
    2. After doing that we have successfully removed some unwanted columns that leads to improve the accuracy somewhat.
5. Splitting the data into train, test with x and y labels:
    1. After doing the above operations I do the train test split by using the train_test_split funtion from sklearn Library with random_state=42, and test_size=0.2.
    2. This gives us the X_train, X_test, Y_train, and Y_test.
## Clustering the data
    1. After splitting the data and getting the x_train, we divide the data into different cluster by using the k-nearest neighbors.
    2. This clustering is done so that we group the same data into different groups/clusters where the data points are very near to each other and their behaviour is related to each other for thr better prediction.
    3. The right numbers of cluster are selected based on elbow method by using the library kneed
    4. For that, we create a new column in the x_test named as cluster where cluster number for each data is written.
    5. Then we do the training and get the different models for different groups/cluster.
    6. This technique is really helpful to increase the accuracy as for the same data different best models are trainned.
## Hyperparameter Tunning
    1. It is very difficult to trained the best model without choosing the best parameters.
    2. So, for this purpose we do the hyperparameter tunning for all the groups'/clusters' models to find the best model.
    3. After finding the best model, we save the model at directory models.
## Data Prediction
    1. For prediction we follow all these above pipelines except Hyperparameter Tunning.
------------
## Technical Aspect
This project is divided into two part:
1. Training a deep learning and machine learning model using Keras.
2. Building deployed, and hosting a Flask web app on Microsoft Azure( https://health-app-by-vicky.azurewebsites.net ).

## Installation
The Code is written in Python 3.8. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [downloading it](https://github.com/IamVicky90/Plant-Disease-Prediction/archive/main.zip):
```bash
pip install -r requirements.txt
```
## Run
To run the app in a local machine, shoot this command in the project directory:
__Run the follwing command after installing requirements.txt__
```bash
python app.py
```
------------

A short description of the project.

Project Organization
------------
## Directory Tree 
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    |── .github
        |── workflows
            |──ci-cd.yaml
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |── demo.PNG           <- Demo Image that is shown in README.md file
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── __init__.py
    |── src
        ├── __pycache__/
        │   ├── __init__.cpython-37.pyc
        │   └── __init__.cpython-38.pyc
        ├── Cluster_Data/
        │   ├── __pycache__/
        │   │   └── cluster.cpython-38.pyc
        │   └── cluster.py
        ├── data/
        │   ├── .gitkeep
        │   ├── __init__.py
        │   └── make_dataset.py
        ├── Data_Preprocessing/
        │   ├── __pycache__/
        │   │   └── preprocessing.cpython-38.pyc
        │   └── preprocessing.py
        ├── log_files/
        ├── Prediction_logs/
        │   ├── Cluster.log
        │   ├── Data_Validation.log
        │   ├── DB_Operations.log
        │   ├── prediction.log
        │   └── preprocessing.log
        └── Training_logs/
            ├── Cluster.log
            ├── Data_Validation.log
            ├── DB_Operations.log
            ├── hyperparameter_training.log
            ├── model_operations.log
            ├── preprocessing.log
    └── X_dummy_selected_list_of_features_by_lasso.txt
        
        ├── logger/
        │   ├── __pycache__/
        │   │   ├── logger.cpython-37.pyc
        │   │   └── logger.cpython-38.pyc
        │   └── logger.py
        ├── models/
        │   ├── .gitkeep
        │   ├── __init__.py
        │   ├── predict_model.py
        │   └── train_model.py
        ├── Prediction_Service/
        │   ├── Data_Validation/
        │   │   ├── __pycache__/
        │   │   │   └── Prediction_Data_Validation.cpython-38.pyc
        │   │   └── Prediction_Data_Validation.py
        │   ├── db/
        │   │   ├── __pycache__/
        │   │   │   ├── db_operations_prediction.cpython-38.pyc
        │   │   │   └── db_operations_training.cpython-38.pyc
        │   │   └── db_operations_prediction.py
        │   └── predict_data/
        │       ├── __pycache__/
        │       │   └── prediction.cpython-38.pyc
        │       └── prediction.py
        ├── Training_Service/
        │   ├── Data_Validation/
        │   │   ├── __pycache__/
        │   │   │   ├── Data_Validation.cpython-37.pyc
        │   │   │   └── Data_Validation.cpython-38.pyc
        │   │   └── Data_Validation.py
        │   ├── db/
        │   │   ├── __pycache__/
        │   │   │   ├── db_operations.cpython-37.pyc
        │   │   │   └── db_operations.cpython-38.pyc
        │   │   └── db_operations.py
        │   ├── Hyperparameter/
        │   │   ├── __pycache__/
        │   │   │   └── hyperparameter_tunning.cpython-38.pyc
        │   │   └── hyperparameter_tunning.py
        │   ├── metrics/
        │   │   ├── __pycache__/
        │   │   │   └── metrics.cpython-38.pyc
        │   │   └── metrics.py
        │   ├── model_operations/
        │   │   ├── __pycache__/
        │   │   │   ├── model_ops.cpython-38.pyc
        │   │   │   └── selecting_the_model.cpython-38.pyc
        │   │   └── model_ops.py
        │   └── wcss_image.png
        └── visualization/
            ├── .gitkeep
            ├── __init__.py
            └── visualize.py
    ├── Prediction_Batch_Files/
        ├── zomato_16052019_12345.csv
        └── zomato_16052019_204900.csv
    |──templates
        index.html
    |──Training_Batch_Files/
        ├── zomato_16052019_12345.csv
        └── zomato_16052019_204900.csv
    |── Master_Prediction_File/
        └── Zomato_prediction.csv
    |── app.py
    |──dvc.loc
    |── dvc.yaml
    |── LICENSE
    |── MakeFile
    |── params.yaml
    |──README.md
    |── requirements.txt
    |──schema_prediction.py
    |──schema_training.py
    |── zomato.db
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
## Libraries
also mentioned in [requirements.txt](https://github.com/IamVicky90/Health-App/blob/main/requirements.txt)
```
gunicorn
absl-py==0.11.0
astunparse==1.6.3
cachetools==4.2.1
certifi==2020.12.5
chardet==4.0.0
click==7.1.2
Flask==1.1.2
flatbuffers==1.12
gast==0.3.3
google-auth==1.25.0
google-auth-oauthlib==0.4.2
google-pasta==0.2.0
grpcio==1.32.0
h5py==2.10.0
idna==2.10
itsdangerous==1.1.0
Jinja2==2.11.3
joblib==1.0.0
jsonify==0.5
Keras-Preprocessing==1.1.2
Markdown==3.3.3
MarkupSafe==1.1.1
numpy==1.19.5
oauthlib==3.1.0
opt-einsum==3.3.0
pandas==1.2.1
protobuf==3.14.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
python-dateutil==2.8.1
pytz==2021.1
requests==2.25.1
requests-oauthlib==1.3.0
rsa==4.7
scikit-learn==0.24.1
scipy==1.6.0
six==1.15.0
sklearn==0.0
tensorboard==2.4.1
tensorboard-plugin-wit==1.8.0
tensorflow==2.4.1
tensorflow-estimator==2.4.0
termcolor==1.1.0
threadpoolctl==2.1.0
typing-extensions==3.7.4.3
urllib3==1.26.3
Werkzeug==1.0.1
wincertstore==0.2
wrapt==1.12.1
xgboost==0.90
```
