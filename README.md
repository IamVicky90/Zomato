Zomato
==============================
# Steps That I take to Solve this ML Problem
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
* In this stage we will do the data preprocessing that includes cleaning the data/columns, EDA(Exploratory Data Analysis),feature engineering, removing or imputing nan values, feature selection and splitting the data into train_test_split.
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
    1. All the Exploratory Data Analysis is done in ![Zomata EDA.ipynb](notebooks/Zomata EDA.ipynb) along with the proper comment.
    2. It is recommended to see this jupyter notebook file, I have explained all the steps beautifully here.
4. Feature Selection:
    1. As there is a lot of features present so that may cause a curse of dimenstionality, to remove this I selected features using Lasso by SelectFromModel funtion from sklearn library.
    2. After doing that we have successfully removed some unwanted columns that leads to improve the accuracy somewhat.
5. Splitting the data into train, test with x and y labels:
    1. After doing the above operations I do the train test split by using the train_test_split funtion from sklearn Library with random_state=42, and test_size=0.2.
    2. This gives us the X_train, X_test, Y_train, and Y_test.

    

------------
A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
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
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
