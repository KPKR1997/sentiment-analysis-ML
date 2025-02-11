import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataingestionConfig
from src.components.model_training import ModelTrainingConfig
from src.components.model_training import Modeltraining
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataingestionConfig:
    csv_path = os.path.join('data', 'raw.csv')
    data_path = os.path.join('artifacts', 'data.csv')
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataingestionConfig()
    
    def initiate_dataIngestion(self):
        logging.info("Data ingestion started")
        try:
            dataframe = pd.read_csv(self.ingestion_config.csv_path, encoding='ISO-8859-1')
            dataframe.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
            dataframe.drop(columns = ['id','date', 'flag', 'user'], inplace=True)
            dataframe.dropna(how='all', inplace=True)
            dataframe.to_csv(self.ingestion_config.data_path)
            logging.info('Raw data processed. Data created in artifacts folder')
            
            logging.info('Initializing train test split')
            train, test = train_test_split(dataframe, test_size = 0.2, random_state=40)
            train.to_csv(self.ingestion_config.train_path, index=False, header = True)
            test.to_csv(self.ingestion_config.test_path, index=False, header = True)
            logging.info('Created train and test data in artifacts')
            

        except Exception as e:
            raise CustomException(e,sys)
        

#

if __name__ == "__main__":
    dataingest = DataIngestion()
    dataingest.initiate_dataIngestion()

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(paths=DataingestionConfig())

    trainer = Modeltraining()
    model = trainer.initiate_model_training(X_train, y_train)
    accuracy = trainer.evaluate_model(model, X_test, y_test)

    trainer.save_model()


#