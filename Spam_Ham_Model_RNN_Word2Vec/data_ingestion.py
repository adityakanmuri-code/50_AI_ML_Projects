import os
import sys
from Spam_Ham_Model_RNN_Word2Vec.exception import CustomException
import Spam_Ham_Model_RNN_Word2Vec.logger as logger
import logging
import pandas as pd
from Spam_Ham_Model_RNN_Word2Vec.config.configuration import Config
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

class DataIngestion:

    def __init__(self,data_type = 'train'):
        try:
            self.config = Config()

            #Load config values
            self.source_type = self.config.get('data',data_type,'source_type')

            if self.source_type == 'file':
                self.file_path = self.config.get('data',data_type,'file_name')
                self.read_params = self.config.get('data',data_type,'read_params')
            elif self.source_type == "scrape":
                self.search_urls = self.config.get('data',data_type,'search_urls')
                self.output_file = self.config.get('data',data_type,'output_file')
                self.max_reviews = self.config.get('data',data_type,'max_reviews')
            elif self.source_type == "kaggle":
                self.data_set_path = self.config.get('data',data_type,'dataset_path')
                self.download_path = self.config.get('data',data_type,'download_path')
                self.new_file_name = self.config.get('data',data_type,'file_name')
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

    def __ingest_kaggle_data(self):
        try:
            #Loading the env file
            load_dotenv()
            #Reading the token value
            token = os.getenv('KAGGLE_API_TOKEN')
            #Creating a new directory for storing the dataset
            os.makedirs(self.download_path,exist_ok=True)
            
            #Initiating new api object to set KaggleApi operations
            api = KaggleApi()
            api.authenticate()

            #Downloading the dataset 
            api.dataset_download_files(
                dataset=self.data_set_path,
                path = self.download_path,
                unzip= True
            )

            csv_files = [
                f for f in os.listdir(self.download_path) 
                if f.endswith("csv")
            ]

            if len(csv_files) == 0:
                error_message = f'No csv files are found in location : {self.download_path}'
                logging.error(error_message)
                raise Exception(error_message)

            source_file = os.path.join(self.download_path,csv_files[0])
            target_file = os.path.join(self.download_path,self.new_file_name)

            os.rename(source_file,target_file)

            return pd.read_csv(target_file)

        except Exception as e:
            raise CustomException(e,sys)

    def ingest_data(self):
        try:
            logging.info('--------------------------------------------------')
            logging.info(f'Data Ingestion has started for {self.source_type}')
            logging.info('--------------------------------------------------')
            if self.source_type == 'file':
                dataframe = self.__ingest_file()
            elif self.source_type == "kaggle":
                dataframe = self.__ingest_kaggle_data()
            logging.info('----------------------------------------------------')
            logging.info(f'Data Ingestion has completed for {self.source_type}')
            logging.info('----------------------------------------------------')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)