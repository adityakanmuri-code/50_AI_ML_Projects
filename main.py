
from Spam_Ham_Model_RNN_Word2Vec.data_ingestion import DataIngestion



def main():
    ingestion = DataIngestion(data_type='train')
    spam_df = ingestion.ingest_data()
    print(spam_df.head())

if __name__ == "__main__":
    main()
