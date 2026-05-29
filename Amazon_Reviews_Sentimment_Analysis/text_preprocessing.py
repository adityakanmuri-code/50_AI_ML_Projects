import os
import sys
from Amazon_Reviews_Sentimment_Analysis.exception import CustomException
import Amazon_Reviews_Sentimment_Analysis.logger as logger
import logging
import warnings
warnings.filterwarnings('ignore')
from Amazon_Reviews_Sentimment_Analysis.config.configuration import Config

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
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
            logging.info('______________________________')
            logging.info('Text Preprocessing has started')
            logging.info('______________________________')
            return dataframe            
        except Exception as e:
            raise CustomException(e,sys)
    
    def lower_text(self,dataframe:pd.DataFrame = None,col:object = None):
        try:
            logging.info(f"Changing the text to lower case in the column : {col}")
            dataframe[col] = dataframe[col].str.lower()
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def remove_special_characters(self,dataframe:pd.DataFrame = None,col:str = None,search_pattern:str=None):
        try:
            logging.info(f"Removing the special characters in the column : {col}")
            dataframe[col] = dataframe[col].apply(lambda x : re.sub(search_pattern," ",str(x)))
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def remove_white_spaces(self,dataframe:pd.DataFrame = None,col:str = None,search_pattern:str = None):
        try:
            logging.info(f"Removing the white spaces in the column : {col}")
            dataframe[col] = dataframe[col].apply(
                lambda x : re.sub(search_pattern,' ',str(x))
            )
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def handling_negations(self,dataframe:pd.DataFrame = None,col:str = None,search_pattern:str = None,replace_pattern:str = None):
        try:
            logging.info(f"Removing the white spaces in the column : {col}")
            dataframe[col] = dataframe[col].apply(
                lambda x : re.sub(search_pattern,replace_pattern,str(x))
            )
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def removing_emoji(self,dataframe:pd.DataFrame = None,col:str = None):
        try:
            logging.info(f"Removing the emojis in the column : {col}")
            dataframe[col] = dataframe[col].apply(
                lambda x : emoji.demojize(x)
            )
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def removing_short_words(self,dataframe:pd.DataFrame = None,col:str = None,word_length:int = 2):
        try:
            logging.info(f"Removing the word less than {word_length} in the column : {col}")
            dataframe[col] = dataframe[col].apply(
                lambda x : ' '.join([word for word in x.split() if len(word) > word_length])
            )
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def handling_url(self,dataframe:pd.DataFrame = None,col:str = None,url_pattern:str = None,html_pattern:str = None):
        try:
            logging.info(f"Removing the urls and https in the column : {col}")
            dataframe[col] = dataframe[col].apply(
                lambda x : re.sub(url_pattern,' ',str(x))
            ).apply(
                lambda x : re.sub(html_pattern,' ',str(x))
            )
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

class Tokenize:
    def __init__(self):
        self.config = Config()
        self.tokenize = self.config.get('training','tokenize')

    def _tokenize_corpus(self,X_train:object = None):
        try:
            logging.info('_____________________')
            logging.info('Tokenizing the corpus')
            logging.info('_____________________')
            documents = X_train.tolist()
            stop_words = stopwords.words(self.tokenize.get('stop_words'))
            corpus = []
            lemma = WordNetLemmatizer()
            for i in range(len(documents)):
                documents[i] = word_tokenize(documents[i])
                documents[i] = [lemma.lemmatize(word) for word in documents[i] if word not in stop_words]
                corpus.append(documents[i])
            logging.info('________________________________________')
            logging.info('Tokenizing the corpus has been completed')
            logging.info('________________________________________')
            return(corpus) 
        except Exception as e:
            raise CustomException(e,sys)
        
class Embedding:
    def _train_generate_embeddings(self,X_train:object = None,vector_size:int = 0,epochs:int = 0,window:int = 0):
        try:
            logging.info("____________________________________________")
            logging.info("Generating the embeddings for Word2Vec Model")
            logging.info("____________________________________________")
            word2vec_model = Word2Vec(X_train,vector_size=vector_size,epochs=epochs,window=window)
            X_train_transformed = []
            for i in range(len(X_train)):
                tokens = X_train[i]
                vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv.key_to_index]
                if len(vectors) == 0:
                    avg_vector = np.zeros(word2vec_model.vector_size)
                else:
                    avg_vector = np.mean(vectors,axis=0)
                X_train_transformed.append(avg_vector)
            logging.info("_______________________________________________________________")
            logging.info("Generating the embeddings for Word2Vec Model has been completed")
            logging.info("_______________________________________________________________")
            return np.array(X_train_transformed),word2vec_model 
        except Exception as e:
            raise CustomException(e,sys)
    
    def _generate_embeddings(self,corpus:object = None,model:object = None):
        try:
            logging.info("____________________________________________")
            logging.info("Generating embeddings using pretrained model")
            logging.info("____________________________________________")
            corpus_transformed = []
            for i in range(len(corpus)):
                tokens = corpus[i]
                vectors = [model.wv[word] for word in tokens if word in model.wv]
                if len(vectors) == 0:
                    avg_vector = np.zeros(model.vector_size)
                else:
                    avg_vector = np.mean(vectors,axis=0)
                corpus_transformed.append(avg_vector)
            logging.info("_______________________________________________________________")
            logging.info("Generating embeddings using pretrained model has been completed")
            logging.info("_______________________________________________________________")
            return corpus_transformed
        except Exception as e:
            raise CustomException(e,sys)