
from Amazon_Reviews_Sentimment_Analysis.data_ingestion import DataIngestion


def main():
    ingestion = DataIngestion(data_type='train')
    senti_df = ingestion.ingest_data()

if __name__ == "__main__":
    main()
