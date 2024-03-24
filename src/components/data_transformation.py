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
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['age', 'bmi', 'children']
            categorical_columns = ['sex', 'smoker', 'region']
            num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
              ])

            cat_pipeline = Pipeline(steps=[
            ("onehot", OneHotEncoder())
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
            
            target_column_name="expenses"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Available columns in train_df: {}".format(input_feature_train_df.columns.tolist()))
            logging.info("Available columns in test_df: {}".format(input_feature_test_df.columns.tolist()))

            logging.info("Column names after dropping:")

            logging.info("target_Train data column: {}".format(train_df['expenses']))
            logging.info("target_Test data column: {}".format(test_df['expenses']))

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )
            # Fit preprocessing object on training data
            preprocessing_obj.fit(input_feature_train_df)

            # Transform both training and testing data using the fitted preprocessing object
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            input_feature_train_df
            # Ensure that target variables are reshaped if necessary
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

          

            # Concatenate input features and target variable for both training and testing data
            train_arr = np.hstack((input_feature_train_arr, target_feature_train_arr))
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_arr))
            
            # Verify shapes after concatenation
            print("\nShapes after concatenation:")
            print("Train array shape:", train_arr.shape)
            print("Test array shape:", test_arr.shape)

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("train_arr data column: {}".format(train_arr.shape))
            logging.info("test_arr data column: {}".format(test_arr.shape))
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise customeException(e, sys)