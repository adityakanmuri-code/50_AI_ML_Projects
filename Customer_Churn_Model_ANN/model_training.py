import os
import sys
from exception import CustomException
import logger
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from config.configuration import Config

import pandas as pd
import numpy as np
import random
from Customer_Churn_Model_ANN.transformer_factory import Transformer_Factory
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras_tuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import pickle

class Model_Trainer:

    def __init__(self,dataframe:pd.DataFrame = None):
        try:
            self.config = Config()
            self.dataframe = dataframe

            self.problem_type = self.config.get("training","problem_type")
            self.target_col = self.config.get("training","target_column")

            self.split_config = self.config.get("training","train_test_split")
            self.preprocessing_config = self.config.get("training","preprocessing")
            self.output_config = self.config.get("training","model_output")

            self.model_config = self.config.get("training","model")
            self.random_seed = self.model_config.get("random_seed")
            self.compile_config = self.model_config.get("compile",{})
            self.callbacks_config = self.compile_config.get("callbacks",[])
            self.hyperparams_config = self.config.get("training","model","hyperparameters")

        except Exception as e:
            raise CustomException(e,sys)
    
    def model_trainer(self):
        try:
            X_train,X_test,y_train,y_test = self.__split_train_test_data(self.dataframe,self.target_col)
            X_train_transformed,preprocessor = self.__fit_transform_preprocess(X_train)

            X_test_transformed = self.__transform_preprocess(X_test,preprocessor)
            self.__dump_model(
                preprocessor,
                file_path=self.output_config.get('base_dir'),
                file_name='preprocessor'
                )
            
            tensorflow.keras.utils.set_random_seed(self.random_seed)
            ann_model = self.__fit_model(X_train=X_train_transformed,X_test=X_test_transformed,y_train=y_train,y_test=y_test)
            model_output_path = self.output_config.get("base_dir")
            model_extension = self.output_config.get("extension")

            self.__dump_model(
                model= ann_model,
                file_path= model_output_path,
                file_name= 'dl_model',
                extension= model_extension
            )

            tensorboard_path = self.output_config.get("model_logs")

        except Exception as e:
            raise CustomException(e,sys)
    
    def __get_optimizer(self,name,params):
        try:
            optimizers = {
                "adam" : tensorflow.keras.optimizers.Adam,
                "sgd" : tensorflow.keras.optimizers.SGD,
                "rmsprop" : tensorflow.keras.optimizers.RMSprop
            }

            if name not in optimizers:
                raise ValueError(f'Invalid Optimizer Name {name}')
                logging.info(f'ERROR : Invalid Optimizer Name {name}')

            return optimizers[name](**params)
        except Exception as e:
            raise CustomException(e,sys)
        
    def __get_losses(self,name):
        try:
            loss = {
                'binary_crossentropy' : tensorflow.keras.losses.BinaryCrossentropy,
                'categorical_crossentropy' : tensorflow.keras.losses.CategoricalCrossentropy,
                'mse' : tensorflow.keras.losses.MeanSquaredError,
                'huber' : tensorflow.keras.losses.Huber
            }

            if name not in loss:
                raise ValueError(f'Loss function name {name} is not found')
                logging.info(f'ERROR : Loss function name {name} is not found')

            return loss[name]()
        except Exception as e:
            raise CustomException(e,sys)

    def __split_train_test_data(self,dataframe:pd.DataFrame = None,target_col:str = None):
        try:
            test_size = self.split_config.get("test_size",0.2)
            seed = self.split_config.get("random_state",43)
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
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed)
            logging.info('_______________________________________________________________')
            logging.info('Splitting the data into train and test data has been completed.')
            logging.info('_______________________________________________________________')
            return X_train,X_test,y_train,y_test
        except Exception as e:
            raise CustomException(e,sys)

    def __build_transformer(self,transform_steps:list = None):
        try:
            logging.info('_______________________________________________________________')
            logging.info(f'Building the transformer as per the list \n {transform_steps}.')
            logging.info('_______________________________________________________________')
            transformers = []
            for step in transform_steps:
                transformer = Transformer_Factory.get_transformer(step['transformer'],**step.get('params',{}))
                transformers.append(
                    (
                        step['transformer'],
                        transformer,
                        step['columns']
                    )
                )
            logging.info('__________________________________________________________________________________')
            logging.info(f'Building the transformer as per the list \n {transform_steps} has been completed.')
            logging.info('__________________________________________________________________________________')
            return ColumnTransformer(transformers=transformers,remainder='passthrough')
        except Exception as e:
            raise CustomException(e,sys)

    def __fit_transform_preprocess(self,X_train:object = None):
        try:
            transform_steps = self.preprocessing_config.get('steps')
            logging.info('__________________________________________')
            logging.info('Fitting and Transforming on the Train Data')
            logging.info('__________________________________________')
            preprocessor = self.__build_transformer(transform_steps=transform_steps)
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info('______________________________________________________________')
            logging.info('Fitting and Transforming on the Train Data has been completed.')
            logging.info('______________________________________________________________')
            return X_train_transformed,preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def __transform_preprocess(self,X_test:object = None,preprocessor:object = None):
        try:
            logging.info('_____________________________')
            logging.info('Transforming on the Test Data')
            logging.info('_____________________________')
            X_test_transformed = preprocessor.transform(X_test)
            logging.info('_________________________________________________')
            logging.info('Transforming on the Test Data has been completed.')
            logging.info('_________________________________________________')
            return X_test_transformed
        except Exception as e:
            raise CustomException(e,sys)

    def __dump_model(self,model:object = None,file_path:str = None,file_name:str = None,extension:object = None):
        try:
            logging.info('___________________________________________')
            logging.info(f'Generating the pickle file for {file_name}')
            logging.info('___________________________________________')
            os.makedirs(file_path,exist_ok=True)
            if extension is None:
                file_path = os.path.join(file_path,f'{file_name}.pkl')
                with open(file_path,'wb') as f:
                    pickle.dump(model,f)
            else:
                file_path = os.path.join(file_path,f'{file_name}.{extension}')
                model.save(file_path)
            logging.info('____________________________________________________________')
            logging.info(f'Pickle file has been generated and available at {file_path}')
            logging.info('____________________________________________________________')
        except Exception as e:
            raise CustomException(e,sys)
                
    def __initiate_callbacks(self):
        try:       
            callbacks = []
            log_path = self.output_config.get('model_logs')
            for cb in self.callbacks_config:
                name = cb.get("name")
                params = cb.get("params")
                if name == 'TensorBoard':
                    file_path = os.path.join(log_path,datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
                    os.makedirs(file_path,exist_ok=True)
                    callbacks.append(
                        TensorBoard(log_dir=file_path,**params)
                    )
                elif name == 'EarlyStopping':
                    callbacks.append(
                        EarlyStopping(**params)
                    )
            return callbacks    
        except Exception as e:
            raise CustomException(e,sys)

    def __fit_model(self,X_train,y_train,X_test,y_test):
        try:
            early_stopping = self.__initiate_callbacks()
            params_config = self.hyperparams_config.get("params",{})
            tuner = RandomSearch(
                hypermodel = lambda hp : self.__build_model(hp,X_train,params_config),
                objective = params_config.get('objective'),
                max_trials=params_config.get('max_trials'),
                executions_per_trial = params_config.get('executions_per_trial'),
                seed= params_config.get('seed'),
                overwrite = True,
                directory = "C:\\Aditya\\ML_Projects\\Customer_Churn_Model_ANN",
                project_name = 'hyperparameters'
            )
            tuner.search(
                X_train,y_train,
                epochs = params_config.get('epochs'),
                validation_data = (X_test,y_test),
                callbacks = early_stopping
            )
            best_model = tuner.get_best_models(num_models=1)[0]
            best_parameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            print("Best Number of Layers : ", best_parameters.get('num_layers'))
            for i in range(best_parameters.get('num_layers')):
                print(
                        f"Units in Layer {i+1} : ",
                        best_parameters.get(f'units_{i}')
                    )
            print(
                    "Best Learning Rate : ",
                    best_parameters.get('learning_rate')
                )
            return best_model
        except Exception as e:
            raise CustomException(e,sys)
    
    def __build_model(self,hp,X_train,params_config):
        try:
                layers = params_config.get('layers',[])
                nuerons = params_config.get('nuerons',[])
                activation = params_config.get('activation',[])
                optimizer_config = self.compile_config.get('optimizer')
                learning_rate = hp.Choice('learning_rate',optimizer_config.get('params',{}).get('learning_rate',[]))
                optimizer = self.__get_optimizer(optimizer_config.get('name'),{'learning_rate' : learning_rate})
                loss = self.__get_losses(self.compile_config.get('loss')) 
                metrics = self.compile_config.get('metrics',[])
                model = keras.Sequential()
                #Input Layer
                model.add(
                    keras.Input(shape=(X_train.shape[1],))
                )
                for i in range(hp.Int('num_layers',layers.get('min_layers'),layers.get('max_layers'))):
                    model.add(Dense(units=hp.Int('units_'+str(i),min_value=nuerons.get('min_value'),max_value=nuerons.get('max_value'),step=nuerons.get('step')),activation=activation.get('hidden_activation')))
                model.add(Dense(nuerons.get('output_nodes'),activation=activation.get('output_activation')))
                model.compile(
                    optimizer = optimizer,
                    loss = loss,
                    metrics = metrics
                )
                return model
        except Exception as e:
            raise CustomException(e,sys)