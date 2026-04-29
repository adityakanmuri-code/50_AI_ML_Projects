import os
import sys
from exception import CustomException
import logger
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,TargetEncoder

class Transformer_Factory:

    @staticmethod
    def get_transformer(transformer_name,**kwargs):
        try:
            transformers = {
                'StandardScaler' : StandardScaler,
                'OneHotEncoder' : OneHotEncoder,
                'OrdinalEncoder' : OrdinalEncoder,
                'TargetEncoder' : TargetEncoder
            }

            if transformer_name not in transformers:
                error_message = f'{transformer_name} is not in the list of transformers.Use a valid list'
                logging.error(error_message)
                raise AttributeError(error_message,sys)
            return transformers[transformer_name](**kwargs)
        except Exception as e:
            raise CustomException(e,sys)