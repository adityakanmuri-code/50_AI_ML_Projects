import os
import sys
from Spam_Ham_Model_RNN_Word2Vec.exception import CustomException
import Spam_Ham_Model_RNN_Word2Vec.logger as logger
import logging

from Spam_Ham_Model_RNN_Word2Vec.config.configuration import Config
import emoji
import pandas as pd
import re

class DataCleaning:
    
    def __init__(self):
        try:
            self.config = Config()
            self.clean_steps = self.config.get('transformation','clean_steps')
            self.text_preprocess_steps = self.config.get('transformation','text_preprocessing')
            self.transform_steps = self.config.get('transformation','transform_steps')
        except Exception as e:
            raise CustomException(e,sys)
        
    def clean_data(self,dataframe:pd.DataFrame = None):
        try:
            if dataframe is None:
                error_message = 'ERROR : Empty Dataset has been recieved.Check the data source'
                logging.info(error_message)
                raise CustomException(error_message,sys)
            logging.info('_____________________________')
            logging.info('Cleaning pipeline has started')
            logging.info('_____________________________')
            for step in self.clean_steps:
                stepname = step['name']
                stepparams = step['params']
                logging.info(f'Executing the step {stepname} with params {stepparams}')

                if not hasattr(self,stepname):
                    error_message = f'ERROR : {stepname} is not a valid step'
                    logging.info(error_message)
                    raise AttributeError(error_message)
                
                method = getattr(self,stepname)
                dataframe = method(dataframe,**stepparams)

            logging.info('________________________________')
            logging.info('Data cleaning has been completed')
            logging.info('________________________________')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def drop_unwanted_columns(self,dataframe:pd.DataFrame = None,collist:list = None):
        try:
            logging.info("__________________________________________________")
            logging.info(f"Dropping the columns {collist} from the dataframe")
            logging.info("__________________________________________________")
            dataframe = dataframe.drop(columns=collist,errors='ignore')
            logging.info("_____________________________________________________________________")
            logging.info(f"Dropping the columns {collist} from the dataframe has been completed")
            logging.info("_____________________________________________________________________")
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def fill_na(self,dataframe:pd.DataFrame = None,tgtcol:str = None):
        try:
            collist = [col for col in dataframe.columns if col != tgtcol]
            for col in collist:
                logging.info('______________________________')
                logging.info(f'Handling Null Values in {col}')
                logging.info('______________________________')
                na_occ = dataframe[col].isna().mean()
                if na_occ >= 0.05:
                    filler = (
                        dataframe[col].median() if(
                            pd.api.types.is_numeric_dtype(dataframe[col].median()) and not pd.api.types.is_bool_dtype(dataframe[col])
                        ) 
                        else (
                            dataframe[col].mode().iloc[0]
                        )
                    )
                    logging.info(f'Imputing the column {col} with {filler}')
                    dataframe[col] = dataframe[col].fillna(filler,axis=0)
                elif na_occ < 0.05:
                    logging.info(f'Dropping all the rows with null values in column {col}')
                    dataframe = dataframe.dropna(subset=[col],axis=0)
                elif na_occ == 0.00:
                     logging.info(f'Column {col} has no null values.Leaving as it is.')
                logging.info('_________________________________________________')
                logging.info(f'Handling Null Values in {col} has been completed')
                logging.info('_________________________________________________')  
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def text_preprocessing(self,dataframe:pd.DataFrame = None):
        try:
            if dataframe is None:
                errormessage = 'ERROR : Data Frame is recieved is empty.Check the data sources'
                logging.info(errormessage)
                raise CustomException(errormessage,sys)
            logging.info("____________________________________________")
            logging.info("Text Preprocessing Pipeline has been started")
            logging.info("____________________________________________")
            for step in self.text_preprocess_steps:
                step_name = step["name"]
                step_params = step['params']
                logging.info(f'{step_name} has started with {step_params}')
                if not hasattr(self,step_name):
                    errormessage = f'ERROR : {step_name} is not a valid step.Check the text preprocessing steps'
                    logging.info(errormessage)
                    raise AttributeError(errormessage,sys)
                method = getattr(self,step_name)
                dataframe = method(dataframe,**step_params)
            logging.info("______________________________________________")
            logging.info("Text Preprocessing pipeline has been completed")
            logging.info("______________________________________________")
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
        
    def lower_case(self,dataframe:pd.DataFrame = None,collist:list = None):
        try:
            logging.info("Converting the text to lower values")
            for col in collist:
                dataframe[col] = dataframe[col].str.lower()
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
        
    def handle_emoji(self,dataframe:pd.DataFrame = None,collist:list = None):
        try:
            logging.info("Handling Emojis in the text")
            for col in collist:
                dataframe[col] = dataframe[col].apply(lambda x : emoji.replace_emoji(x,replace=' '))
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def handle_special_chars(self,dataframe:pd.DataFrame = None,collist:list = None,search_pattern:str = None):
        try:
            logging.info('Handling Special Characters')
            for col in collist:
                dataframe[col] = dataframe[col].apply(lambda x : re.sub(search_pattern,' ',str(x)))
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def remove_white_spaces(self,dataframe:pd.DataFrame = None,collist:list = None,search_pattern:str = None):
        try:
            for col in collist:
                logging.info(f"Removing the white spaces in the column : {col}")
                dataframe[col] = dataframe[col].apply(
                    lambda x : re.sub(search_pattern,' ',str(x))
                )
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def handling_shortwords(self,dataframe:pd.DataFrame = None,collist:list = None,word_length:int = 2):
        try:
            for col in collist:
                logging.info(f'Removing shortwords in the column {col}')
                dataframe[col] = dataframe[col].apply(
                    lambda x : ' '.join([word for word in x.split() if len(word) > word_length])
                )
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
        
    def remove_business_prefix(self,dataframe:pd.DataFrame = None,collist:list = None,search_pattern:str= None):
        try:
            for col in collist:
                logging.info(f'Removing Business Prefix from the {col}')
                dataframe[col] = dataframe[col].apply(lambda x : re.sub(search_pattern,' ',str(x)))
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def transform_data(self,dataframe:pd.DataFrame = None):
        try:
            if dataframe is None:
                errormessage = f'ERROR : Data Frame is empty check the data source'
                logging.info(errormessage)
                raise CustomException(errormessage,sys)
            logging.info("__________________________________________________________________________________________")
            logging.info(f"Data Transformation Pipeline has been started with following steps {self.transform_steps}")
            logging.info("__________________________________________________________________________________________")
            for step in self.transform_steps:
                step_name = step['name']
                step_params = step['params']

                if not hasattr(self,step_name):
                    errormessage = f'ERROR : Invalid step name {step_name} in the available steps'
                    logging.info(errormessage)
                    raise AttributeError(errormessage,sys)
                method = getattr(self,step_name)
                dataframe = method(dataframe,**step_params)
            logging.info("____________________________________________________________________________________________")
            logging.info(f"Data Transformation Pipeline has been completed with following steps {self.transform_steps}")
            logging.info("____________________________________________________________________________________________")
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def merge_columns(self,dataframe:pd.DataFrame = None,column1:str = None,column2:str = None,target_col:str = None):
        try:
            logging.info(f"Merging the columns {column1} and {column2} into {target_col}")
            dataframe[target_col] = dataframe[column1] + " " + dataframe[column2]
            #Removing the columns that have been merged
            dataframe = dataframe.drop(columns=[column1,column2],errors='ignore')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
