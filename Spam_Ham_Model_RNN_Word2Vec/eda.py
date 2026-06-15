from dotenv import load_dotenv
import os
import sys
from Spam_Ham_Model_RNN_Word2Vec.exception import CustomException
import Spam_Ham_Model_RNN_Word2Vec.logger as logger
import logging
from Spam_Ham_Model_RNN_Word2Vec.config.configuration import Config

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

class EDA:
    def __init__(self):
        try:
            self.config = Config()
            self.base_folder = self.config.get('eda','base_folder')
            self.dpi = self.config.get('eda','dpi')
            self.bbox_inches = self.config.get('eda','bbox_inches')
            self.extension = self.config.get('eda','extension')
            self.target_col = self.config.get('eda','target_col')
            self.plot_steps = self.config.get('eda','plot_steps')
            self.chisquare_words = self.config.get('eda','spam_words_chi_square_scores','word_count')
            self.chisquare_fname = self.config.get('eda','spam_words_chi_square_scores','file_name')
            self.heatmap_fname = self.config.get('eda','spam_ham_percentage_comparison','file_name')
            self.chiscore_dist_bins = self.config.get('eda','chi_score_distribution','bins')
            self.chiscore_dist_fname = self.config.get('eda','chi_score_distribution','file_name')
            self.hamspam_cluster_model = self.config.get('eda','spam_ham_clusters','model_name')
            self.hamspam_cluster_n_comp = self.config.get('eda','spam_ham_clusters','n_components')
            self.hamspam_cluster_random_state = self.config.get('eda','spam_ham_clusters','random_state')
            self.hamspam_cluster_fname = self.config.get('eda','spam_ham_clusters','file_name')
        except Exception as e:
            raise CustomException(e,sys)
    
    def data_analysis(self,dataframe:pd.DataFrame = None):
        try:
            if dataframe is None:
                error_message = 'ERROR : DataFrame is empty check the data source for more understanding'
                logging.info(error_message)
                raise CustomException(error_message,sys)
            for step in self.plot_steps:
                step_name = step['name']
                step_params = step['params'] or {}
                if not hasattr(self,step_name):
                    error_message = f'ERROR : Step Name {step_name} is not a valid step'
                    logging.info(error_message)
                    raise AttributeError(error_message,sys)
                method = getattr(self,step_name)
                method(dataframe,**step_params)
        except Exception as e:
            raise CustomException(e,sys)

    def text_label_analysis(self,dataframe:pd.DataFrame = None,text_column:str = None,stopwords:str = None,min__df:int = 5,word_count:int = 0):
        try:
            vectorizer = CountVectorizer(stop_words = stopwords,min_df=min__df,ngram_range=(1,2))
            X = vectorizer.fit_transform(dataframe[text_column])
            chi_scores,_ = chi2(X,dataframe[self.target_col])
            top_words = [
                vectorizer.get_feature_names_out()[i] 
                for i in chi_scores.argsort()[-word_count:][::-1]
            ]
            top_words_reversed = []
            for i in range(len(top_words)-1,-1,-1):
                top_words_reversed.append(top_words[i])
            logging.info(f'INFO : Top 20 Spam words are  : {top_words_reversed}')

            n_spam = dataframe.loc[dataframe['label'] == 1].shape[0]
            n_ham = dataframe.loc[dataframe['label'] == 0].shape[0]
            spam_pct = []
            ham_pct = []
            for w in top_words_reversed:
                n_in_spam = dataframe.loc[dataframe['label'] == 1,'text'].str.contains(w,case=False).sum()
                n_in_ham = dataframe.loc[dataframe['label'] == 0,'text'].str.contains(w,case=False).sum()
                spam_pct.append(round(n_in_spam/n_spam,4))
                ham_pct.append(round(n_in_ham/n_ham,4))
            pct_data = pd.DataFrame({
                'Spam %' : spam_pct,
                'Ham %' : ham_pct
            },index = top_words_reversed)
            self.__spam_words_chi_square_scores(chi_scores,vectorizer)
            self.__spam_ham_percentage_comparison(pct_data)
            self.__chi_score_distribution(chi_scores)
            self.__spam_ham_clusters(dataframe)
        except Exception as e:
            raise CustomException(e,sys)
    
    def __spam_words_chi_square_scores(self,chi_scores:object = None,vectorizer:object=None):
        try:
            top_indices = chi_scores.argsort()[-self.chisquare_words:]
            top_words = [
                vectorizer.get_feature_names_out()[i] for i in top_indices
            ]
            top_scores = [
                chi_scores[i] for i in top_indices
            ]
            plt.figure(figsize=(10,6))
            plt.barh(top_words,top_scores)
            plt.xlabel('Chi-Square Scores')
            plt.ylabel('Words')
            plt.title(f'Top {self.chisquare_words} Spam Predictive Words')
            plt.tight_layout()
            self.__save_plots(self.chisquare_fname)
        except Exception as e:
            raise CustomException(e,sys)
    
    def __spam_ham_percentage_comparison(self,df:pd.DataFrame = None):
        try:
            plt.figure(figsize=(10,6))
            sns.heatmap(df,annot=True)
            plt.title('Spam-Ham Percentage Compairson')
            self.__save_plots(self.heatmap_fname)
        except Exception as e:
            raise CustomException(e,sys)

    def __chi_score_distribution(self,chi_scores:object = None):
        try:
            plt.figure(figsize=(10,6))
            plt.hist(chi_scores,bins = self.chiscore_dist_bins)
            plt.xlabel('Chi-Square Scores')
            plt.ylabel('Number of Features')
            plt.title('Distribution of Chi Scores')
            self.__save_plots(self.chiscore_dist_fname)
        except Exception as e:
            raise CustomException(e,sys)

    def __spam_ham_clusters(self,dataframe:pd.DataFrame = None):
        try:
            load_dotenv()
            hf_token = os.getenv('HUGGING_FACE_API_TOKEN')
            model = SentenceTransformer(self.hamspam_cluster_model,token=hf_token)
            embeddings = model.encode(dataframe['text'].tolist())

            tsne = TSNE(n_components=self.hamspam_cluster_n_comp,random_state=self.hamspam_cluster_random_state)
            data_2d = tsne.fit_transform(embeddings)

            plt.figure(figsize=(20,10))
            plt.scatter(data_2d[dataframe['label']==1, 0], data_2d[dataframe['label']==1, 1], 
                c='red', alpha=0.5, label='Spam', s=10)
            plt.scatter(data_2d[dataframe['label']==0, 0], data_2d[dataframe['label']==0, 1], 
                c='blue', alpha=0.5, label='Ham', s=10)
            plt.legend()
            plt.title('Spam Ham Clusters')
            self.__save_plots(self.hamspam_cluster_fname)
        except Exception as e:
            raise CustomException(e,sys)

    def __save_plots(self,file_name:str = None):
        try:
            logging.info('___________________________________________________________________')
            logging.info(f'Plots for column {file_name} are being saved at {self.base_folder}')
            logging.info('___________________________________________________________________')
            file_path = ''
            timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
            os.makedirs(self.base_folder,exist_ok=True)
            
            file_path = os.path.join(
                self.base_folder,
                f"{file_name}_{timestamp}.{self.extension}"
            )

            plt.savefig(file_path,dpi=self.dpi,bbox_inches = self.bbox_inches)
            plt.close()
            logging.info('____________________________________________')
            logging.info(f'Plot is available at the {self.base_folder}')
            logging.info('____________________________________________')
        except Exception as e:
            raise CustomException(e,sys)