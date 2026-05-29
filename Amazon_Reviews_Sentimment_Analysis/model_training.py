import os
import sys
from Amazon_Reviews_Sentimment_Analysis.exception import CustomException
import Amazon_Reviews_Sentimment_Analysis.logger as logger
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from Amazon_Reviews_Sentimment_Analysis.config.configuration import Config
from Amazon_Reviews_Sentimment_Analysis.text_preprocessing import Embedding,Tokenize
from imblearn.over_sampling import RandomOverSampler

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

class Model_Trainer:

    def __init__(self,dataframe:pd.DataFrame = None):
        try:
            self.config = Config()
            self.embeddings = Embedding()
            self.dataframe = dataframe

            self.problem_type = self.config.get("training","problem_type")
            self.target_col = self.config.get("training","target_column")
            self.split_config = self.config.get("training","split")
            self.embedding_config = self.config.get("training","embeddings")
            self.model_ouput = self.config.get("training","model_output")

        except Exception as e:
            raise CustomException(e,sys)
    
    def model_trainer(self):
        try:
            X_train,X_test,y_train,y_test = self.__split_train_test_data(self.dataframe,self.target_col)
            #Before Tokenizing remember to convert the X_train from pd.DataFrame to pd.Series
            tokenizer = Tokenize()
            X_train = tokenizer._tokenize_corpus(X_train.iloc[:,0])
            X_train_transformed,embed_model = self.__fit_transform_preprocess(X_train)
            self.__dump_model(model=embed_model,file_path= self.model_ouput['model_dir'],file_name='w2vmodel',extension=self.model_ouput['embed_extension'])
            #Reducing the imbalances in the train data
            X_train_resampled,y_train_resampled = self.__reduce_imbalances(X_train=X_train_transformed,y_train=y_train)
            X_test = tokenizer._tokenize_corpus(X_test.iloc[:,0])
            X_test_transformed = self.__transform_preprocess(X_test,embed_model)
            #Training the model
            model,model_name = self.__train_model(X_train_resampled,y_train_resampled,X_test_transformed,y_test)
            self.__dump_model(model=model,file_path= self.model_ouput['model_dir'],file_name='predict_model',extension=self.model_ouput['model_extension'])
        except Exception as e:
            raise CustomException(e,sys)
    
    def __split_train_test_data(self,dataframe:pd.DataFrame = None,target_col:str = None):
        try:
            test_size = self.split_config.get("test_size")
            seed = self.split_config.get("random_state")
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
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed,stratify=y)
            logging.info('_______________________________________________________________')
            logging.info('Splitting the data into train and test data has been completed.')
            logging.info('_______________________________________________________________')
            return X_train,X_test,y_train,y_test
        except Exception as e:
            raise CustomException(e,sys)

    def __fit_transform_preprocess(self,X_train:object = None):
        try:
            logging.info('__________________________________________')
            logging.info('Fitting and Transforming on the Train Data')
            logging.info('__________________________________________')
            X_train_transformed,embed = self.embeddings._train_generate_embeddings(X_train,vector_size = self.embedding_config['vector_size'],epochs=self.embedding_config['epochs'],window=self.embedding_config['window']) 
            logging.info('______________________________________________________________')
            logging.info('Fitting and Transforming on the Train Data has been completed.')
            logging.info('______________________________________________________________')
            return X_train_transformed,embed
        except Exception as e:
            raise CustomException(e,sys)
    
    def __transform_preprocess(self,X_test:object = None,model:object = None):
        try:
            logging.info('_____________________________')
            logging.info('Transforming on the Test Data')
            logging.info('_____________________________')
            X_test_transformed = self.embeddings._generate_embeddings(X_test,model)
            logging.info('_________________________________________________')
            logging.info('Transforming on the Test Data has been completed.')
            logging.info('_________________________________________________')
            return X_test_transformed
        except Exception as e:
            raise CustomException(e,sys)

    def __reduce_imbalances(self,X_train:object = None,y_train:object = None):
        try:
            logging.info("___________________________________")
            logging.info("Reducing the imbalances in the data")
            logging.info("___________________________________")
            ros = RandomOverSampler(random_state=self.split_config['random_state'])
            logging.info("______________________________________________________")
            logging.info("Reducing the imbalances in the data has been completed")
            logging.info("______________________________________________________")
            return ros.fit_resample(X_train,y_train)
        except Exception as e:
            raise CustomException(e,sys)

    def __train_model(self,X_train,y_train,X_test,y_test):
        try:
            logging.info("____________________________________")
            logging.info("Training the model on the train data")
            logging.info("____________________________________")
            best_score = -1
            models = {
                'LogisticRegression' : LogisticRegression(),
                'DecisionTreeClassifier' : DecisionTreeClassifier(),
                'RandomForestClassifier' : RandomForestClassifier(),
                'GaussianNB' : GaussianNB(),
                'LinearSVC' : LinearSVC()
            }
            for name,model in models.items():
                model.fit(X_train,y_train)
                scores = self.__model_eval(model,X_test,y_test)
                if scores['F1_Score'] > best_score:
                    best_model = model
                    best_model_name = model.__class__.__name__
                    best_score = scores['F1_Score']
            logging.info("_______________________________________________________")
            logging.info("Training the model on the train data has been completed")
            logging.info("_______________________________________________________")
            return best_model,best_model_name
        except Exception as e:
            raise CustomException(e,sys)

    def __model_eval(self,model,X_test,y_test):
        try:
            y_pred = model.predict(X_test)
            logging.info("__________________________________________________")
            logging.info(f'Evaluation Metrics for {model.__class__.__name__}')
            logging.info(f'Accuracy : {accuracy_score(y_test,y_pred)}')
            logging.info(f'Precision_Score : {precision_score(y_test,y_pred)}')
            logging.info(f'Recall_Score : {recall_score(y_test,y_pred)}')
            logging.info(f'F1_Score : {f1_score(y_test,y_pred)}')
            logging.info("__________________________________________________")
            return{
                'Model_Name': model.__class__.__name__,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1_Score': f1_score(y_test, y_pred,average='weighted'),
                'Precision_Score' : precision_score(y_test,y_pred),
                'Recall_Score' : recall_score(y_test,y_pred),
            }
        except Exception as e:
            raise CustomException(e,sys)

    def __dump_model(self,model:object = None,file_path:str = None,file_name:str = None,extension:object = None):
        try:
            logging.info('___________________________________________')
            logging.info(f'Generating the pickle file for {file_name}')
            logging.info('___________________________________________')
            os.makedirs(file_path,exist_ok=True)
            if extension == 'pkl':
                file_path = os.path.join(file_path,f'{file_name}.pkl')
                with open(file_path,'wb') as f:
                    pickle.dump(model,f)
            elif extension == 'h5':
                file_path = os.path.join(file_path,f'{file_name}.{extension}')
                model.save(file_path)
            elif extension == 'model':
                file_path = os.path.join(file_path,f'{file_name}.{extension}')
                model.save(file_path)
            logging.info('____________________________________________________________')
            logging.info(f'Pickle file has been generated and available at {file_path}')
            logging.info('____________________________________________________________')
        except Exception as e:
            raise CustomException(e,sys)
            