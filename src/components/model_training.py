import os
import sys
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass
from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from src.components.data_ingestion import data



@dataclass
class ModelTrainingConfig:
    model_path = os.path.join('artifacts', 'model.pkl')

class Modeltraining:
    def __init__(self):
        self.modelpath = ModelTrainingConfig()

    def initiate_model_training(self, X_train, Y_train):
        logging.info("Model training initiated")
        try:
            model = LogisticRegression(max_iter=1500)
            model.fit(X_train, Y_train)
            logging.info('Model traing completed')
            return model
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def evaluate_model(self, model, X_test, Y_test):
        logging.info("started model evaluation")
        try:
            Y_pred = model.predict(X_test)
            accuracy = accuracy_score(Y_pred, Y_test)
            return accuracy
        
        except Exception as e:
            raise CustomException(e,sys)

    def save_model(self, model):
        try:
            save_obj(self.modelpath, model)
            logging.info("Saved trained model in artifacts successfully")
        except Exception as e:
            raise CustomException(e,sys)
        
