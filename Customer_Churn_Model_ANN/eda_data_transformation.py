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
import seaborn as sns
import matplotlib.pyplot as plt 
from Customer_Churn_Model_ANN.data_ingestion import DataIngestion

class CleanData:
    def __init__(self):
        self.config = Config()
        self.clean_steps = self.config.get("transformation","clean_steps")
        self.transform_steps = self.config.get("transformation","transform_steps")

    def clean_data(self,dataframe:pd.DataFrame = None):
        try:
            if dataframe is None:
                raise ValueError('Input dataframe cannot be None')
            logging.info('Cleaning pipeline started')

            for step in self.clean_steps:
                step_name = step['name']
                step_params = step.get('params',{})

                logging.info(f'Executing step: {step_name} with params: {step_params}')

                if not hasattr(self,step_name):
                    error_message = f'Cleaning step has no attr {step_name}'
                    logging.error(error_message)
                    raise AttributeError(error_message,sys)
                
                method = getattr(self,step_name)

                dataframe = method(dataframe,**step_params)

            logging.info("Cleaning pipeline completed")

            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def drop_unwanted_cols(self,dataframe:pd.DataFrame = None , drop_col:list = None):
        try:
            logging.info('---------------------------------------')
            logging.info(f'Dropping unwanted columns : {drop_col}.')
            logging.info('---------------------------------------')
            dataframe = dataframe.drop(columns=drop_col,errors = 'ignore')
            logging.info('-----------------------------------------------------------')
            logging.info(f'Dropping unwanted columns : {drop_col} has been completed.')
            logging.info('-----------------------------------------------------------')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def fill_na(self,dataframe:pd.DataFrame = None,targetcol:str = None):
        try:
            collist = [c for c in dataframe.columns if c!= targetcol]
            logging.info('---------------------')
            logging.info('Imputing Missing Data')
            logging.info('---------------------')
            for col in collist:
                nan_occ = dataframe[col].isna().mean()
                if nan_occ >= 0.05:
                    filler = (
                        dataframe[col].median() if (
                            pd.api.types.is_numeric_dtype(dataframe[col]) and not pd.api.types.is_bool_dtype(dataframe[col])
                        )
                        else (
                            dataframe[col].mode().iloc[0]
                        )
                    )
                    logging.info(f'Imputing the data in {col} with {filler}')
                    dataframe[col] = dataframe[col].fillna(filler,axis=0)
                elif nan_occ > 0.05:
                    logging.info(f'Dropping the rows in {col} with null values since missing data is {nan_occ*100}% which less than threshold 5%')
                    dataframe = dataframe.dropna(subset=[col],axis=0)
                elif nan_occ == 0.0:
                    logging.info(f'No imputing required since there is no missing data')
            return dataframe
        except Exception as e :
            raise CustomException(e,sys)

    def transform_data(self,dataframe:pd.DataFrame = None):
        try:
            logging.info('-------------------------------------------')
            logging.info('Data Transformation has started.')
            logging.info('-------------------------------------------')

            for step in self.transform_steps:
                step_name = step['name']
                step_params = step.get('params',{})

                if not hasattr(self,step_name):
                    error_message = f'{step_name} is not a valid step name.Kindly use a valid step name'
                    logging.error(error_message)
                    raise AttributeError(error_message,sys)
                
                method = getattr(self,step_name)
                dataframe = method(dataframe,**step_params)
            logging.info('--------------------------------------')
            logging.info('Data transformation has been completed')
            logging.info('--------------------------------------')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def convert_dtypes(self,dataframe:pd.DataFrame = None,desired_dtype:str = None,collist:list = None):
        try:
            logging.info('----------------------------------------------------------')
            logging.info(f'Converting the datatype for {collist} to {desired_dtype}.')
            logging.info('----------------------------------------------------------')
            for col in collist:
                dataframe[col] = dataframe[col].astype(desired_dtype)
            logging.info('---------------------------------------')
            logging.info(f'Datatype conversion has been completed')
            logging.info('---------------------------------------')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def encode_category(self,dataframe:pd.DataFrame = None,mappings=None):
        try:
            logging.info('-------------------------------------------------------------------------------')
            logging.info(f'Encode the categorical data in {mappings.items()}')
            logging.info('-------------------------------------------------------------------------------')
            
            for col,mapval in mappings.items():
                if col not in dataframe.columns:
                    raise ValueError(f'{col} not found in dataframe')
                dataframe[col] = dataframe[col].map(mapval).fillna(dataframe[col])

            logging.info('-----------------------------------------------')
            logging.info('Encodng the categorical data has been completed')
            logging.info('-----------------------------------------------')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def strip_spaces(self,dataframe:pd.DataFrame = None,collist:list = None):
        try:
            logging.info('---------------------------------------------')
            logging.info(f'Stripping white spaces for {collist} columns')
            logging.info('---------------------------------------------')
            self.collist = collist
            for col in self.collist:
                logging.info(f'Stripping the white space for {col}')
                dataframe[col] = dataframe[col].str.strip()
            logging.info('---------------------------------------------')
            logging.info('Stripping the white spaces has been completed')
            logging.info('---------------------------------------------')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def fix_outliers(self,dataframe:pd.DataFrame = None,numcols:list = None,fix_mode:str = 'IQR',fix_type:str = 'clip',threshold:float = 1.5):
        try:
            for col in numcols:
                logging.info('__________________________________________________________________________________')
                logging.info(f'Fixing the outliers in the column {col} using {fix_mode} and appliying {fix_type}')
                logging.info('__________________________________________________________________________________')
                if fix_mode.upper() == 'IQR':
                    Q1,Q3,IQR = self.__calculate_IQR(dataframe=dataframe,col=col)
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                elif fix_mode.upper() == 'ZSCORE':
                    mean,std = self.__calculate_zscore(dataframe=dataframe,col=col)
                    lower = mean - threshold * std
                    upper = mean + threshold * std
                else:
                    error_message = f'Value Error : {fix_mode} is not a valid method to fix outliers.'
                    logging.error(error_message)
                    raise ValueError(error_message,sys)
                outlier_mask = (dataframe[col] < lower) | (dataframe[col] > upper)
                outlier_count = outlier_mask.sum()
                if outlier_count == 0:
                    logging.info(f'Outlier Count for {col} is {outlier_count}.No fixing is needed.')
                    continue
                if fix_type.upper() == 'CLIP':
                    dataframe[col] = dataframe[col].clip(lower,upper)
                elif fix_type.upper() == 'DROP':
                    dataframe = dataframe.loc[~outlier_mask]
                else:
                    error_message = f'Value Error : {fix_type} is not a valid method.'
                    logging.error(error_message)
                    raise ValueError(error_message,sys)
                logging.info('_____________________________________________________________________________________________________')
                logging.info(f'Fixing the outliers in the column {col} using {fix_mode} and appliying {fix_type} has been completed')
                logging.info('_____________________________________________________________________________________________________')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def __calculate_IQR(self,dataframe:pd.DataFrame=None,col:str=None):
        try:
            logging.info(f'Calculating the IQR for {col}')
            Q1 = dataframe[col].quantile(0.25)
            Q3 = dataframe[col].quantile(0.75)
            IQR = Q3-Q1
            logging.info(f'IQR for {col} is {IQR}')
            return Q1,Q3,IQR
        except Exception as e:
            raise CustomException(e,sys)

    def __calculate_zscore(self,dataframe:pd.DataFrame=None,col:str = None):
        try:
            logging.info(f'Calculating ZScore for {col}.')
            mean = dataframe[col].mean()
            std = dataframe[col].std()
            logging.info(f'Mean and Std Deviation for {col} is {mean},{std} respectively.')
            return mean,std
        except Exception as e:
            raise CustomException(e,sys)
    
    def clean_column_names(self,dataframe:pd.DataFrame = None):
        try:
            logging.info('___________________________________________')
            logging.info('Removing white spaces from the column names')
            logging.info('___________________________________________')
            dataframe.columns = dataframe.columns.str.strip().str.lower()
            logging.info('___________________________________________')
            logging.info('Removing white spaces from the column names')
            logging.info('___________________________________________')
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

class EDA:
    def __init__(self,dataframe:pd.DataFrame = None):
        try:
            self.config = Config()
            self.dataframe = dataframe

            self.plotsteps = self.config.get('eda','plot_data')

            self.extension = 'png'
            self.dpi = 400
        except Exception as e:
            raise CustomException(e,sys)

    def __get_base_path(self,base_folder,subfolder):
        timestamp = datetime.now().strftime('%d_%m_%Y')
        path = os.path.join(base_folder,subfolder,timestamp)
        os.makedirs(path,exist_ok=True)
        return path

    def plot_data(self):
        try:
            for step in self.plotsteps:
                step_name = step['name']
                step_params = step.get('params',{})

                if not hasattr(self,step['name']):
                    error_message = f"{step['name']} is not valid step name"
                    logging.error(error_message)
                    raise AttributeError(error_message,sys)
                
                method = getattr(self,step_name)
                method(
                    **step_params
                )
        except Exception as e:
            raise CustomException(e,sys)

    def cat_dist_plot(self,base_folder:str = None,catcols:list = None):
        try:
            logging.info('----------------------------------------------------------------------------------------------------------')
            logging.info(f'Plotting the categorical distribution of {catcols}. The plots will be available in path {base_folder}')
            logging.info('----------------------------------------------------------------------------------------------------------')
            for col in catcols:
                break
                col_count = self.dataframe[col].value_counts()
                plt.figure(figsize=(10,8))
                sns.barplot(x=col_count.index,y=col_count.values,legend = 'brief',width=0.4)
                plt.title(f'Distribution of {col.upper()}')
                plt.xlabel(col)
                plt.ylabel('Distribution')
                file_name = f'Distribution of {col.upper()}'
                self.save_plots(base_folder=base_folder,file_name=file_name)
            logging.info('--------------------------------------------------------')
            logging.info('Plotting for categorical distribution has been completed')
            logging.info('--------------------------------------------------------')
        except Exception as e:
            raise CustomException(e,sys)

    def num_dist_plot(self,base_folder:str = None,numcols:list = None):
        try:
            base_folder = self.__get_base_path(base_folder,'Distribution_Numerical_Data')
            logging.info('----------------------------------------------------------------------------------------------------')
            logging.info(f'Plotting the numerical distribution for {numcols}. The plots will be available in {base_folder}')
            logging.info('----------------------------------------------------------------------------------------------------')
            for col in numcols:
                plt.figure(figsize=(10,8))
                plt.title(f'Distribution of {col.upper()}')
                sns.histplot(data=self.dataframe,x=col,bins=30,kde=True)
                plt.xlabel(col.upper())
                plt.ylabel('Distribution')
                file_name = f'Distribution of {col.upper()}'
                self.save_plots(base_folder,file_name)
            logging.info('--------------------------------------------------------')
            logging.info('Plotting for numerical distribution has been completed')
            logging.info('--------------------------------------------------------')
        except Exception as e:
            raise CustomException(e,sys)

    def cat_tgt_plot(self,base_folder:str = None,catcols:list = None,tgtcol:str = None):
        try:
            base_folder = self.__get_base_path(base_folder,'Relational_Categorical_Target_Data')
            for col in catcols:
                logging.info('---------------------------------------------------------------------')
                logging.info(f'Plotting the distribution between {col.upper()} and {tgtcol.upper()}')
                logging.info('---------------------------------------------------------------------')
                col_count = self.dataframe.groupby(col)[tgtcol].value_counts().reset_index(name='Count')
                plt.figure(figsize=(10,8))
                sns.barplot(data=col_count,x=col,y='Count',hue=tgtcol,width=0.5)
                plt.title(f'{col.upper()} Vs {tgtcol.upper()}')
                plt.xlabel(col.upper())
                plt.ylabel('Distribution')
                file_name = f'{col.upper()} Vs {tgtcol.upper()}'
                logging.info('----------------------------------------------------------------------------------------')
                logging.info(f'Plotting the distribution between {col.upper()} and {tgtcol.upper()} has been completed')
                logging.info('----------------------------------------------------------------------------------------')
                self.save_plots(base_folder=base_folder,file_name=file_name)
        except Exception as e:
            raise CustomException(e,sys)

    def outliers_plot(self,base_folder:str = None,numcols:list = None,tgtcol:str = None):
        try:
            base_folder = self.__get_base_path(base_folder,'Outliers_Plot')
            for col in numcols:
                plt.figure(figsize=(10,8))
                sns.boxplot(data = self.dataframe,y=col,hue = tgtcol,width=0.4)
                plt.title(f'Outliers in {col.upper()}')
                plt.xlabel('Outliers')
                plt.ylabel(col.upper())
                file_name = f'Outliers in {col.upper()}'
                self.save_plots(base_folder = base_folder,file_name = file_name)
        except Exception as e:
            raise CustomException(e,sys)

    def correlation_plot(self,base_folder:str = None):
        try:
            base_folder = self.__get_base_path(base_folder,"Correlation")

            plt.figure(figsize=(12,10))
            corr = self.dataframe.corr(numeric_only=True)
            sns.heatmap(corr,annot=True,cmap='coolwarm')
            plt.title("Correlation Heatmap")
            self.save_plots(base_folder,"Correlation_Heatmap")
        except Exception as e:
            raise CustomException(e,sys)

    def save_plots(self,base_folder:str = None,file_name:str = None):
        try:
            logging.info('--------------------------------------------------------------')
            logging.info(f'Plots for column {file_name} are being saved at {base_folder}')
            logging.info('--------------------------------------------------------------')
            file_path = ''
            timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
            os.makedirs(base_folder,exist_ok=True)
            
            file_path = os.path.join(
                base_folder,
                f"{file_name}_{timestamp}.{self.extension}"
            )

            plt.savefig(file_path,dpi=self.dpi,bbox_inches = 'tight')
            plt.close()
            logging.info('---------------------------------------')
            logging.info(f'Plot is available at the {base_folder}')
            logging.info('---------------------------------------')
        except Exception as e:
            raise CustomException(e,sys)