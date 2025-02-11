import os
import sys
import pickle
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj
from src.components.data_transformation import stemming

class ModelPredictor:
    def __init__(self):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            self.model = load_obj(model_path)
            logging.info("Loaded trained model successfully")

            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            self.preprocessor = load_obj(preprocessor_path)
            logging.info("Loaded preprocessor successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def predict_sentiment(self, text):
        try:
            logging.info(f"Predicting sentiment for: {text}")

            stemmed_text = stemming(text)
            vectorized_text = self.preprocessor.transform([stemmed_text])

            prediction = self.model.predict(vectorized_text)

            sentiment = "Positive comment" if prediction[0] == 1 else "Negative comment"

            return sentiment

        except Exception as e:
            raise CustomException(e, sys)
