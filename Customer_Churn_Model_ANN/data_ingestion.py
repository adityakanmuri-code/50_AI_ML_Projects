import os
import sys
from exception import CustomException
import logger
import logging
import pandas as pd

class DataIngestion:

    def __init__(self,source_type):
        self.source_type = source_type
    
    def __ingest_file(self,*args,**kwargs):
        try:
            if len(args) == 0:
                error_message = f'Invalid args.Kindly provide args that will be needed to run the file ingestion'
                logging.error(error_message)
                raise AttributeError(error_message,sys)
            file_path = args[0]
            if file_path.endswith('csv'):
                logging.info(f'File has been read from {file_path}')
                return pd.read_csv(file_path,**kwargs)
            elif file_path.endswith('xlsx'):
                return pd.read_excel(file_path,**kwargs)
            elif file_path.endswith('json'):
                return pd.read_json(file_path,**kwargs)
            elif file_path.endswith('parquet'):
                return pd.read_parquet(file_path,**kwargs)
            else:
                _,extension =  os.path.splitext(file_path)
                extension = extension[1:]
                error_message = f'Invalid file extension {extension} kindly provide a file with valid extension'
                logging.error(error_message)
                raise AttributeError(error_message,sys)
        except Exception as e:
            raise CustomException(e,sys)
    
    def ingest_data(self,*args,**kwargs):
        try:
            logging.info('--------------------------------------------------')
            logging.info(f'Data Ingestion has started for {self.source_type}')
            logging.info('--------------------------------------------------')
            if self.source_type == 'file':
                dataframe = self.__ingest_file(*args,**kwargs)
            logging.info('----------------------------------------------------')
            logging.info(f'Data Ingestion has completed for {self.source_type}')
            logging.info('----------------------------------------------------')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)