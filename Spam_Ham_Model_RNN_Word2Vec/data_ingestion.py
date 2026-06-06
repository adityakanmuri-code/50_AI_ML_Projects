from dotenv import load_dotenv
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

load_dotenv()
token= os.getenv("KAGGLE_API_TOKEN")

dataset_path = "ssssws/spam-email-detection-dataset-clean-and-ml-ready"

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    dataset=dataset_path,
    path="C:\\Aditya\\ML_Practice\\ML_Projects\\Spam_Ham_Model_RNN_Word2Vec\\datasets",
    unzip=True
)
