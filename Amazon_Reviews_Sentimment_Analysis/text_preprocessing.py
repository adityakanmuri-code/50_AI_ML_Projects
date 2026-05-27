import os
import sys
from Amazon_Reviews_Sentimment_Analysis.exception import CustomException
import Amazon_Reviews_Sentimment_Analysis.logger as logger
import logging
import warnings
warnings.filterwarnings('ignore')
from Amazon_Reviews_Sentimment_Analysis.config.configuration import Config

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import Word2Vec
import re
import pandas as pd
import emoji

class Text_Preprocessing:
    def __init__(self):
        self.config = Config()
        self.text_preprocess_steps = self.config.get('transformation','text_preprocessing','preprocess_steps')

    def preprocess_text(self,dataframe:pd.DataFrame = None):
        try:
            logging.info('______________________________')
            logging.info('Text Preprocessing has started')
            logging.info('______________________________')
            for step in self.text_preprocess_steps:
                step_name = step['name']
                step_params = step.get('params',{})
                if not hasattr(self,step_name):
                    error_message = f'{step_name} is not a valid step'
                    logging.info(error_message)
                    raise AttributeError(error_message)
                method = getattr(self,step_name)
                dataframe = method(dataframe,**step_params)
        except Exception as e:
            raise CustomException(e,sys)
    
    def lower_text(self,dataframe:pd.DataFrame = None,col:str = None):
        try:
            dataframe[col] = dataframe[col].str.lower()
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def remove_special_characters(self,dataframe:pd.DataFrame = None,col:str = None,search_pattern:str=None):
        try:
            dataframe[col] = dataframe[col].apply(lambda x : re.sub(search_pattern," ",str(x)))
            print(dataframe[col].head())
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
