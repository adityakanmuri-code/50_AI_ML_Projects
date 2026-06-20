
from Spam_Ham_Model_RNN_Word2Vec.data_ingestion import DataIngestion
from Spam_Ham_Model_RNN_Word2Vec.data_cleaning import DataCleaning
from Spam_Ham_Model_RNN_Word2Vec.eda import EDA
from Spam_Ham_Model_RNN_Word2Vec.model_training import Trainer



def main():
    ingestion = DataIngestion(data_type='train')
    spam_df = ingestion.ingest_data()
    cleaner = DataCleaning()
    spam_df = cleaner.clean_data(spam_df)
    spam_df = cleaner.text_preprocessing(spam_df)
    spam_df = cleaner.transform_data(spam_df)
    #eda = EDA()
    #eda.data_analysis(spam_df)
    trainer = Trainer()
    trainer.model_trainer(spam_df)
    

if __name__ == "__main__":
    main()
