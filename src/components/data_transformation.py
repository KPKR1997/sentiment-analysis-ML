import os
import sys
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer


from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from src.components.data_ingestion import DataingestionConfig
from src.components.data_ingestion import DataIngestion

from dataclasses import dataclass


def stemming(data):
    stemmer = PorterStemmer()
    data = re.sub('[^a-zA-Z]', ' ', data)
    data = data.lower().split()
    data = [stemmer.stem(word) for word in data if word not in stopwords.words('english')]
    return ' '.join(data)


@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor(self):
        try:
            logging.info('Attempting data transformation')

            pipeline = Pipeline([
                ('vectorization', TfidfVectorizer(stop_words='english'))
            ])

            preprocessor = ColumnTransformer(
                [
                    ('text', pipeline, 'stemmed_text')
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    

    def initiate_data_transformation(self, paths = DataingestionConfig()):

        logging.info('Initiating data transformation on dataframe')

        try:
            train_data = pd.read_csv(paths.train_path)
            test_data = pd.read_csv(paths.test_path)

            train_data['stemmed_text'] = train_data['text'].apply(stemming)
            test_data['stemmed_text'] = test_data['text'].apply(stemming)

            feature_train = train_data['stemmed_text']
            feature_test = test_data['stemmed_text']
            target_train = train_data['target']
            target_test = test_data['target']

            

            target_train.replace(4, 1, inplace=True)
            target_test.replace(4, 1, inplace=True)

            feature_train = feature_train.apply(stemming)
            feature_test = feature_test.apply(stemming)

            preprocessor_obj = self.get_preprocessor()

            feature_train_arr = preprocessor_obj.fit_transform(feature_train)
            feature_test_arr = preprocessor_obj.transform(feature_test)



            logging.info("preprocessing completed")

            save_obj(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessor_obj
            )
            logging.info('transformation preprocessor file created')
            return(
                feature_train_arr,
                feature_test_arr,
                target_train,
                target_test
            )



        except Exception as e:
            raise CustomException(e,sys)
        



