import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from xgboost import XGBRegressor

from src.exception import customeException

from src.logger import logging

from src.utils import save_object,evaluatemodels

@dataclass

class Modeltrainingconfig:
    model_training_path:str = os.path.join("artifact","model.pkl")
    
class modeltraining:
    def __init__(self):
        self.modeltraining_config=Modeltrainingconfig()
    def initiate_model_trainig(self,train_array, test_array):
        try:
            
            logging.info("Split training and test data from model")
            # Splitting train_array into features (X) and target variable (y)
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
        
            # Splitting test_array into features (X) and target variable (y)
            x_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            
            # Reshaping data
            y_train = np.reshape(y_train, (-1, 1))
            y_test = np.reshape(y_test, (-1, 1))
            

            models={
                "randomForest":RandomForestRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "Linearregressor":LinearRegression(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting": CatBoostRegressor(verbose=False),
                "Ada Bossting": AdaBoostRegressor()
            }
            
            model_report:dict=evaluatemodels(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            
            #To get best model score form dict
            best_model_score= max(sorted(model_report.values()))
            #to get best model name
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            logging.info("best Model Name:{}".format(best_model_name))
            
            if best_model_score < 0.6 :
                raise customeException("No best model found")
            logging.info("Best Model found on traing and test datasets")
            
            save_object (
                 file_path=self.modeltraining_config.model_training_path,
                 obj=best_model
                    )
            predicted=best_model.predict(x_test)
            
            r2=r2_score(y_test,predicted)
            
            return r2
             
        except Exception as e:
            raise customeException(e,sys)
            
