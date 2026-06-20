import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
from Spam_Ham_Model_RNN_Word2Vec.exception import CustomException
import Spam_Ham_Model_RNN_Word2Vec.logger as logger
import logging
from Spam_Ham_Model_RNN_Word2Vec.config.configuration import Config

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats import chisquare
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.layers import TextVectorization,Input,Embedding,SimpleRNN,Dense,Dropout,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

class Trainer:
    def __init__(self):
        self.config = Config()
        self.split_params = self.config.get('model','split_train_test')
        self.p_threshold = self.config.get('model','oversample','p_threshold')
        self.tokenize_params = self.config.get('model','tokenize_text')
        self.glove_params = self.config.get('model','glove_dictionary')
        self.output_params = self.config.get('model','model_output')
        self.word_coverage_threshold = self.config.get('model','word_coverage_threshold')
        self.encode_data = self.config.get('model','encode_data')

    def model_trainer(self,dataframe:pd.DataFrame = None):
        try:
            logging.info('___________________________________')
            logging.info('Model Training Pipeline has started')
            logging.info('___________________________________')
            X_train,X_test,y_train,y_test = self.__split_train_test(dataframe,train_size = self.split_params['train_size'],random_state = self.split_params['random_state'],stratify_flag = self.split_params['stratify_flag'],target_col=self.split_params['target_col'])
            
            #Seperating text and categorical columns
            X_train_text = X_train[self.tokenize_params['txt_column']].astype(str).values
            X_test_text = X_test[self.tokenize_params['txt_column']].astype(str).values
            X_train_cat = X_train.drop(columns=self.tokenize_params['txt_column'])
            X_test_cat = X_test.drop(columns=self.tokenize_params['txt_column'])

            #Encoding Categorical Data
            X_train_cat,ohe = self.__encode_data(X_train_cat,data_type='train')
            X_test_cat = self.__encode_data(X_test_cat,data_type='test',transformer=ohe)

            #Generating Embedding Layer
            X_train_text_tokenized,tokenizer = self.__tokenize_text(data_type='train',text=X_train_text)
            X_test_text_tokenized = self.__tokenize_text(data_type='test',text=X_test_text,vectorizer=tokenizer)
            glove_dict = self.__build_glove_dictionary()
            embed_matrix,word_index,word_vocab = self.__build_embed_matrix(glove_dict,tokenizer)
            word_coverage = self.__validate_coverage(glove_dict,word_index)
            if word_coverage >= self.word_coverage_threshold:
                self.__dump_model(embed_matrix,self.output_params['embed_matrix_fname'],self.output_params['embed_matrix_extn'])
                self.__dump_model(word_vocab,self.output_params['vocab_fname'],self.output_params['model_extn'])
                self.__dump_model(tokenizer,self.output_params['text_vectorizer_fname'],self.output_params['model_extn'])
                self.__dump_model(word_index,self.output_params['word_index_fname'],self.output_params['model_extn'])
            logging.info('__________________________________________')
            logging.info('Model Training Pipeline has been completed')
            logging.info('__________________________________________')
        except Exception as e:
            raise CustomException(e,sys)
    
    def __split_train_test(self,df:pd.DataFrame = None,train_size:float = None,random_state:int = None,stratify_flag:bool = False,target_col:str = ''):
        try:
            logging.info('___________________________________________________________________________________________________________________')
            logging.info(f'Splitting Train Test Data with train size {train_size},seed value {random_state} and stratified is {stratify_flag}')
            logging.info('___________________________________________________________________________________________________________________')
            #Splitting the features and labels
            collist = [col for col in df.columns if col != target_col]
            X = df[collist]
            y = df[target_col]
            if stratify_flag == True:
                X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=train_size,random_state=random_state,stratify=y)
                return X_train,X_test,y_train,y_test
            elif stratify_flag == False:
                X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=train_size,random_state=random_state)
                return X_train,X_test,y_train,y_test
        except Exception as e:
            raise CustomException(e,sys)

    def __oversample_data(self,X_train:object = None,y_train:object = None,df:pd.DataFrame = None,p_threshold:float = 0.0):
        try:
            logging.info('______________________________________________________')
            logging.info('Reducing the imbalances in data based on chiscore test')
            logging.info('______________________________________________________')
            observed_counts = df['label'].value_counts().values
            _,p_value = chisquare(f_obs= observed_counts)
            if p_value < 0.05:
                smote = SMOTE(random_state=self.split_params['random_state'])
                X_train_resampled,y_train_resampled = smote.fit_resample(X_train,y_train)
                logging.info('Data has been balanced now')
                return X_train_resampled,y_train_resampled
            else:
                logging.info('Since the threshold is more no rebalncing is required')
                return X_train,y_train
        except Exception as e:
            raise CustomException(e,sys)
    
    def __tokenize_text(self,data_type:str = '',text:object = None,vectorizer:object = None):
        try:
            #Converting the pandas to text since tensorflow expects data in 1 dimensional array but X_train and X_test are in pandas
            if data_type == 'train':
                vectorizer = TextVectorization(
                    max_tokens = self.tokenize_params['max_vocab_size'],
                    output_mode = self.tokenize_params['output_mode'],
                    standardize = self.tokenize_params['standardize'],
                    output_sequence_length = self.tokenize_params['max_seq_length']
                )
                vectorizer.adapt(text)
                text_tokenized = vectorizer(text)
                return text_tokenized,vectorizer
            elif data_type == 'test':
                if vectorizer is None:
                    error_message = f'ERROR : Vectorizer is missing.Kindly check if the vectorizer has been passed'
                    logging.info(error_message)
                    raise CustomException(error_message,sys)
                text_tokenized = vectorizer(text)
                return text_tokenized
        except Exception as e:
            raise CustomException(e,sys)

    def __build_glove_dictionary(self):
        try:
            logging.info('Generating the GloVe dictionary')
            embedding_index = {}
            glove_file = os.path.join(self.glove_params['glove_path'],self.glove_params['glove_file_name'])
            with open(file=glove_file,encoding=self.glove_params['encoding'],mode='r') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if (not re.search(r'[^\w\s]',word)) and (len(word) > 2) :
                        coefs = np.asarray(values[1:],dtype = np.float32)
                        embedding_index[word] = coefs
            return(embedding_index)
        except Exception as e:
            raise CustomException(e,sys)    

    def __build_embed_matrix(self,glove_dict:object = None,vectorizer:object = None):
        try:
            logging.info('Generating Word Index')
            vocab = vectorizer.get_vocabulary()
            vocab_size = len(vocab)
            word_index = {
                word : idx for idx,word in enumerate(vocab)
            }
            logging.info('Building Embedding Matrix')
            embedding_matrix = np.zeros(
                (
                    vocab_size,
                    self.glove_params['embedding_dim']
                )
            )
            for word,idx in word_index.items():
                embedding_vector = glove_dict.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
            return embedding_matrix,word_index,vocab
        except Exception as e:
            raise CustomException(e,sys)
        
    def __validate_coverage(self,glove_dict:object = None,word_index:object = None):
        try:
            logging.info('Validating the Coverage for the words')
            hits = 0
            misses = 0
            for word in word_index:
                if word in glove_dict:
                    hits+=1
                else:
                    misses+=1
            coverage = hits/(hits+misses)
            return int(coverage*100)
        except Exception as e:
            raise CustomException(e,sys)

    def __encode_data(self,X:object=None,data_type:str = '',transformer:object=None):
        try:
            logging.info('Encoding the data')
            if data_type == 'train':
                ohe = OneHotEncoder(handle_unknown=self.encode_data['handle_unknown'],sparse_output=self.encode_data['sparse_output'])
                ct = ColumnTransformer(
                    [
                        ('OneHotEncoder',ohe,self.encode_data['collist'])
                    ]
                )
                X_encode = ct.fit_transform(X)
                X_encode = pd.DataFrame(
                    X_encode,
                    columns = ct.get_feature_names_out(),
                    index = X.index
                )
                X = X.drop(columns=self.encode_data['collist'])
                X = pd.concat([X,X_encode],axis=1)
                return X,ct
            elif data_type == 'test':
                if transformer is None:
                    errormessage = f'ERROR : Transformer is None check the function call'
                    logging.info(errormessage)
                    raise CustomException(errormessage,sys)
                X_encode = transformer.transform(X[self.encode_data['collist']])
                X_encode = pd.DataFrame(
                    X_encode,
                    columns = transformer.get_feature_names_out(),
                    index = X.index
                )
                X = X.drop(columns=self.encode_data['collist'])
                X = pd.concat([X,X_encode],axis=1)
                return X
        except Exception as e:
            raise CustomException(e,sys)

    def __build_model(self,vocab:object = None,embedding_matrix:object = None,X_encode:object = None):
        try:
            #Create Text Input Branch
            text_input = Input(
                shape = (self.tokenize_params['max_seq_length'],),
                name = 'Text_Input_Layer'
            )
            #Creating Embedding Layer
            embedding_layer = Embedding(
                input_dim = len(vocab),
                output_dim= self.glove_params['embedding_dim'],
                weights= [embedding_matrix],
                input_length=self.tokenize_params['max_seq_length'],
                trainable=False
            )(text_input)

            #Adding Simple RNN Layer
            rnn_branch = SimpleRNN(
                units = 128,
                activation = 'tanh'
            )(embedding_layer)

            #Adding Dense Layer for Text Features
            rnn_branch = Dense(
                64,
                activation = 'relu'
            )(
                rnn_branch
            )
            rnn_branch = Dropout(
                0.3
            )(rnn_branch)
            
            #Creating Categorical Branch
            n_cat_features = X_encode.shape[1] 
            cat_input = Input(
                shape = (n_cat_features,),
                name = 'Categorical_Input_Layer'
            )

            #Category Dense Layers
            cat_branch = Dense(
                64,
                activation='relu'
            )(
                cat_input
            )
            cat_branch = Dropout(
                0.3
            )(
                cat_branch
            )
            #Merging Text and Categorical Data
            merged = Concatenate()(
                [
                    rnn_branch,
                    cat_branch
                ]
            )
            #Final Classification Layer
            merged = Dense(
                64,
                activation = 'relu'
            )(
                merged
            )
            merged = Dropout(
                0.3
            )(
                merged
            )
            merged = Dense(
                32,
                activation = 'relu'
            )(
                merged
            )
            #Output Layer
            output = Dense(
                1,
                activation = 'sigmoid'
            )(
                merged
            )

            #Create Model
            model = Model(
                inputs = [
                    text_input,
                    cat_input
                ],
                outputs = output
            )
        except Exception as e:
            raise CustomException(e,sys)

    def __dump_model(self,model:object = None,file_name:str = None,extension:str = None):
        try:
            os.makedirs(self.output_params['base_path'],exist_ok=True)
            file_path = os.path.join(self.output_params['base_path'],f'{file_name}.{extension}')
            if extension == 'npy':
                np.save(
                    file_path,
                    model
                )
            elif extension == 'pkl':
                with open (file_path,'wb') as f:
                    pickle.dump(model,f)
            elif extension == 'h5' :
                model.save(file_path)
        except Exception as e:
            raise CustomException(e,sys)