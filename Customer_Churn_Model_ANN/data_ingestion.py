import os
import sys
from exception import CustomException
import logger
import logging
import pandas as pd
from config.configuration import Config

class DataIngestion:

    def __init__(self,data_type = 'train'):
        try:
            self.config = Config()

            #Load config values
            self.source_type = self.config.get("data",data_type,"source_type")
            self.file_path = self.config.get("data",data_type,"file_path")
            self.read_params = self.config.get("data",data_type,"read_params")
        except:
            self.read_params = {}
        
    
    def __ingest_file(self):
        try:
           if not self.file_path:
               raise ValueError("File path is missing in config")
           logging.info(f"Reading file from {self.file_path}")
           if self.file_path.endswith("csv"):
               return pd.read_csv(self.file_path,**self.read_params)
           elif self.file_path.endswith("xlsx"):
               return pd.read_excel(self.file_path,**self.read_params)
           else:
               _,extension = os.path.splitext(self.file_path)
               extension = extension[1:]
               raise ValueError(f'Unsupported file extension: {extension}')
        except Exception as e:
            raise CustomException(e,sys)
    
    def ingest_data(self):
        try:
            logging.info('--------------------------------------------------')
            logging.info(f'Data Ingestion has started for {self.source_type}')
            logging.info('--------------------------------------------------')
            if self.source_type == 'file':
                dataframe = self.__ingest_file()
            logging.info('----------------------------------------------------')
            logging.info(f'Data Ingestion has completed for {self.source_type}')
            logging.info('----------------------------------------------------')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)