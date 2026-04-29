import os
import sys
from exception import CustomException
import logger
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from config.configuration import Config

import pandas as pd
from Customer_Churn_Model_ANN.transformer_factory import Transformer_Factory
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import pickle

class Model_Trainer:

    def __init__(self,dataframe:pd.DataFrame = None):
        try:
            self.config = Config()
            self.dataframe = dataframe

            self.problem_type = self.config.get("training","problem_type")
            self.target_col = self.config.get("training","target_column")

            self.split_config = self.config.get("training","train_test_split")
            self.transform_steps = self.config.get("training","preprocessing")

        except Exception as e:
            raise CustomException(e,sys)
    
    def model_trainer(self):
        try:
            X_train,X_test,y_train,y_test = self.__split_train_test_data(self.dataframe,self.target_col)
            X_train_transformed,preprocessor = self.__fit_transform_preprocess(X_train)
            feature_names = preprocessor.get_feature_names_out()
            X_test_transformed = self.__transform_preprocess(X_test,preprocessor)
            self.__dump_model(
                preprocessor,
                file_path=self.transform_steps.get('file_path'),
                file_name=self.transform_steps.get('file_name')
                )
        except Exception as e:
            raise CustomException(e,sys)
            
    def __split_train_test_data(self,dataframe:pd.DataFrame = None,target_col:str = None):
        try:
            test_size = self.split_config.get("test_size",0.2)
            seed = self.split_config.get("random_state",43)
            logging.info('____________________________________________')
            logging.info('Splitting the data into features and target.')
            logging.info('____________________________________________')
            collist = [c for c in self.dataframe.columns if c != target_col]
            X = dataframe[collist]
            y = dataframe[target_col]
            logging.info('_______________________________________________________________')
            logging.info('Splitting the data into features and target has been completed.')
            logging.info('_______________________________________________________________')
            logging.info('____________________________________________')
            logging.info('Splitting the data into train and test data.')
            logging.info('____________________________________________')
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed)
            logging.info('_______________________________________________________________')
            logging.info('Splitting the data into train and test data has been completed.')
            logging.info('_______________________________________________________________')
            return X_train,X_test,y_train,y_test
        except Exception as e:
            raise CustomException(e,sys)

    def __build_transformer(self,transform_steps:list = None):
        try:
            logging.info('_______________________________________________________________')
            logging.info(f'Building the transformer as per the list \n {transform_steps}.')
            logging.info('_______________________________________________________________')
            transformers = []
            for step in transform_steps:
                transformer = Transformer_Factory.get_transformer(step['transformer'],**step.get('params',{}))
                transformers.append(
                    (
                        step['transformer'],
                        transformer,
                        step['columns']
                    )
                )
            logging.info('__________________________________________________________________________________')
            logging.info(f'Building the transformer as per the list \n {transform_steps} has been completed.')
            logging.info('__________________________________________________________________________________')
            return ColumnTransformer(transformers=transformers,remainder='passthrough')
        except Exception as e:
            raise CustomException(e,sys)

    def __fit_transform_preprocess(self,X_train:object = None):
        try:
            transform_steps = self.transform_steps.get('steps')
            logging.info('__________________________________________')
            logging.info('Fitting and Transforming on the Train Data')
            logging.info('__________________________________________')
            preprocessor = self.__build_transformer(transform_steps=transform_steps)
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info('______________________________________________________________')
            logging.info('Fitting and Transforming on the Train Data has been completed.')
            logging.info('______________________________________________________________')
            return X_train_transformed,preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def __transform_preprocess(self,X_test:object = None,preprocessor:object = None):
        try:
            logging.info('_____________________________')
            logging.info('Transforming on the Test Data')
            logging.info('_____________________________')
            X_test_transformed = preprocessor.transform(X_test)
            logging.info('_________________________________________________')
            logging.info('Transforming on the Test Data has been completed.')
            logging.info('_________________________________________________')
            return X_test_transformed
        except Exception as e:
            raise CustomException(e,sys)

    def __dump_model(self,model:object = None,file_path:str = None,file_name:str = None):
        try:
            logging.info('___________________________________________')
            logging.info(f'Generating the pickle file for {file_name}')
            logging.info('___________________________________________')
            os.makedirs(file_path,exist_ok=True)
            file_path = os.path.join(file_path,f'{file_name}.pkl')
            with open(file_path,'wb') as f:
                pickle.dump(model,f)
            logging.info('____________________________________________________________')
            logging.info(f'Pickle file has been generated and available at {file_path}')
            logging.info('____________________________________________________________')
        except Exception as e:
            raise CustomException(e,sys)