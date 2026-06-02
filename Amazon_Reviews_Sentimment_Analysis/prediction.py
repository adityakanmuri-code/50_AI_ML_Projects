import os
import sys
from Amazon_Reviews_Sentimment_Analysis.exception import CustomException
import Amazon_Reviews_Sentimment_Analysis.logger as logger
import logging
import warnings
warnings.filterwarnings('ignore')
from Amazon_Reviews_Sentimment_Analysis.config.configuration import Config

from Amazon_Reviews_Sentimment_Analysis.data_ingestion import DataIngestion
from Amazon_Reviews_Sentimment_Analysis.eda_data_transformation import CleanData
from Amazon_Reviews_Sentimment_Analysis.text_preprocessing import Text_Preprocessing,Tokenize,Embedding
import numpy as np
import pickle
from gensim.models import Word2Vec
class Prediction:

    def __init__(self):
        self.config = Config()
        self.predict_params = self.config.get('prediction')
        self.cleaner = CleanData()
        self.text_preprocessor = Text_Preprocessing()
        self.tokenizer = Tokenize()
        self.embedding = Embedding()

    def _predictor(self):
        try:
            scraped_data = DataIngestion(data_type = 'scrape')
            scraped_df = scraped_data.ingest_data()
            
            scraped_df = self.cleaner.clean_data(dataframe=scraped_df)
            scraped_df = self.cleaner.transform_data(dataframe=scraped_df)
            scraped_df = self.text_preprocessor.preprocess_text(dataframe=scraped_df)
            X_pred = scraped_df['reviewText']
            corpus = self.tokenizer._tokenize_corpus(X_pred)
            embed_path = os.path.join(self.predict_params['predict_model_dir'],self.predict_params['embed_model'])
            w2v_model = Word2Vec.load(embed_path)
            X_transformed = self.embedding._generate_embeddings(
                corpus= corpus,
                model = w2v_model
            )
            X_transformed = np.array(X_transformed)
            
            predict_path = os.path.join(self.predict_params['predict_model_dir'],self.predict_params['predict_model'])
            with open(predict_path,'rb') as f:
                predict_model = pickle.load(f)

            predictions = predict_model.predict(X_transformed)

            scraped_df['predicted_sentiment'] = predictions
        except Exception as e:
            raise CustomException(e,sys)