import os
import sys
from exception import CustomException
import logger
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from config.configuration import Config
import pickle
import tensorflow
from sklearn.metrics import accuracy_score

from Customer_Churn_Model_ANN.data_ingestion import DataIngestion
from Customer_Churn_Model_ANN.eda_data_transformation import CleanData
from Customer_Churn_Model_ANN.model_training import Model_Trainer

class Prediction():
    def __init__(self):
        try:
            self.config = Config()
            #Predictions for unseen data
            self.model_path = self.config.get('training','model_output','base_dir')
            self.target_column = self.config.get('training','target_column')
        except Exception as e:
            raise CustomException(e,sys)
    
    def predict_data(self):
        try:
            ingest = DataIngestion(data_type='predict')
            dataframe = ingest.ingest_data()
            
            cleaner = CleanData()
            dataframe = cleaner.clean_data(dataframe=dataframe)
            dataframe = cleaner.transform_data(dataframe=dataframe)

            X = dataframe.loc[:,dataframe.columns != self.target_column]
            y_true = dataframe[self.target_column]
            
            preprocessor = self.__fetch_models(model_type='preprocessor')
            dl_model = self.__fetch_models(model_type='model')
            X_transformed = preprocessor.transform(X)
            y_pred = dl_model.predict(X_transformed)
            y_pred = (y_pred > 0.4).astype(int)

            score = accuracy_score(y_true,y_pred)
            print(score)
        
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def __fetch_models(self,model_type:str = None):
        try:
            if model_type == 'preprocessor':
                file_path = os.path.join(self.model_path,'preprocessor.pkl')
                with open(file_path,'rb') as f:
                    model = pickle.load(f)
            elif model_type == 'model':
                file_path = os.path.join(self.model_path,'dl_model.h5')
                model = tensorflow.keras.models.load_model(file_path)
            return model
        except Exception as e:
            raise CustomException(e,sys)
