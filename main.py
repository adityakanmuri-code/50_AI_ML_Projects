
from Spam_Ham_Model_RNN_Word2Vec.data_ingestion import DataIngestion
from Spam_Ham_Model_RNN_Word2Vec.data_cleaning import DataCleaning
from Spam_Ham_Model_RNN_Word2Vec.eda import EDA
import matplotlib.pyplot as plt



def main():
    ingestion = DataIngestion(data_type='train')
    spam_df = ingestion.ingest_data()
    cleaner = DataCleaning()
    spam_df = cleaner.clean_data(spam_df)
    spam_df = cleaner.text_preprocessing(spam_df)
    spam_df = cleaner.transform_data(spam_df)
    eda = EDA()
    eda.data_analysis(spam_df)
    

if __name__ == "__main__":
    main()
