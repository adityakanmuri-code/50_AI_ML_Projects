import os
import sys
from Amazon_Reviews_Sentimment_Analysis.exception import CustomException
import Amazon_Reviews_Sentimment_Analysis.logger as logger
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from Amazon_Reviews_Sentimment_Analysis.config.configuration import Config
from sklearn.model_selection import GridSearchCV

class Hyperparameter_Tuning:

    def __init__(self):
        self.config = Config()
        self.hyperparameters_tuning = self.config.get('training','hyperparameters')
        self.params = self.config.get('training','hyperparameters','params')

    def _hyperparameter_tuner(self,model,X_train,y_train):
        try:
            model_name = model.__class__.__name__
            if model_name not in self.hyperparameters_tuning:
                error_message = f'{model_name} is not in the hyperparameters'
                logging.warning(error_message)
                raise CustomException(error_message,sys)
            param_grid = self.hyperparameters_tuning[model_name]
            logging.info('________________________________________________')
            logging.info(f'Starting Hyperparameter Tuning for {model_name}')
            logging.info('________________________________________________')
            grid_search = GridSearchCV(
                estimator= model,
                param_grid= param_grid,
                scoring= self.params['scoring'],
                cv = self.params['cv'],
                n_jobs = self.params['n_jobs'],
                verbose = self.params['verbose']
            )
            grid_search.fit(X_train,y_train)

            logging.info('__________________________________________________________________')
            logging.info(f'Best Parameters for {model_name} are : {grid_search.best_params_}')
            logging.info(f'Best Score for {model_name} is : {grid_search.best_score_}')
            logging.info('__________________________________________________________________')

            return grid_search.best_estimator_,grid_search.best_score_
        except Exception as e:
            raise CustomException(e,sys)
