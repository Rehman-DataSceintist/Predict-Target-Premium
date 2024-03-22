import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import customeException
from src.utils import save_object
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['age', 'bmi', 'children', 'expenses']
            categorical_columns = ['sex', 'smoker', 'region']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder())
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor
        
             

        except Exception as e:
            raise customeException(e, sys)
    logging.info("combined columns:{preprocessor}")
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_obj()

            
            
            logging.info("Column names before dropping:")
            logging.info("Train data columns: {}".format(train_df.columns))
            logging.info("Test data columns: {}".format(test_df.columns))
            
            # Print the column names to verify if they exist
            logging.info("Available columns in train_df: {}".format(train_df.columns.tolist()))
            logging.info("Available columns in test_df: {}".format(test_df.columns.tolist()))
            
            input_feature_train_df = train_df.iloc[0:52000]
            target_feature_train_df = train_df['expenses'].iloc[0:52000]
            

            input_feature_test_df = test_df.iloc[0:52000]
            target_feature_test_df = test_df['expenses'].iloc[0:52000]
            
            # Print the column names to verify if they exist
            logging.info("Available columns in train_df: {}".format(input_feature_train_df))
            logging.info("Available columns in test_df: {}".format(input_feature_test_df))
            
            logging.info("Column names after dropping:")
            logging.info("Train data columns: {}".format(input_feature_train_df.columns))
            logging.info("Test data columns: {}".format(input_feature_test_df.columns))
            logging.info("target_Train data column: {}".format(target_feature_test_df))
            logging.info("target_Test data column: {}".format(target_feature_test_df))

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise customeException(e, sys)
