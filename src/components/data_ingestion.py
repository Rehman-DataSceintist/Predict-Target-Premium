import os
import sys
from src.exception import customeException
from src.logger import logging
import pandas as pd
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import Modeltrainingconfig
from src.components.model_trainer import modeltraining
from dataclasses import dataclass

@dataclass
class Data_Ingestion_Config:
    train_data_path:str = os.path.join("artifact",'train.csv')
    test_data_path:str = os.path.join("artifact",'test.csv')
    raw_data_path:str = os.path.join("artifact",'rawdata.csv')

class DataIngestion:
    def __init__(self):
        self.Ingestion_config=Data_Ingestion_Config()
    def initiate_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            df=pd.read_csv('notebook\data\Insurence.csv')
            logging.info("Read the dataset as a data frame")
            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.Ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiate")

            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.Ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.Ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
            )

        except Exception as e:
            raise customeException(e,sys)
        
if __name__=='__main__':
    
    obj= DataIngestion()

    train_path,test_path=obj.initiate_ingestion()
    
    
    data_transformation=DataTransformation()
    
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path,test_path)
    
    obj=modeltraining()
    
    r2=obj.initiate_model_trainig(train_arr,test_arr)
    
    print(r2)
    
    logging.info(f"R-squared score of the best model: {r2}")
    
 


