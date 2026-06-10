
from Spam_Ham_Model_RNN_Word2Vec.data_ingestion import DataIngestion
from Spam_Ham_Model_RNN_Word2Vec.data_cleaning_eda import DataCleaning



def main():
    ingestion = DataIngestion(data_type='train')
    spam_df = ingestion.ingest_data()
    cleaner = DataCleaning()
    spam_df = cleaner.clean_data(spam_df)
    spam_df = cleaner.text_preprocessing(spam_df)
    print(spam_df.head())
    

if __name__ == "__main__":
    main()
