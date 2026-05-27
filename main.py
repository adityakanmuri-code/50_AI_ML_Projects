
from Amazon_Reviews_Sentimment_Analysis.data_ingestion import DataIngestion
from Amazon_Reviews_Sentimment_Analysis.eda_data_transformation import CleanData
from Amazon_Reviews_Sentimment_Analysis.text_preprocessing import Text_Preprocessing


def main():
    ingestion = DataIngestion(data_type='train')
    senti_df = ingestion.ingest_data()

    cleaner = CleanData()
    senti_df = cleaner.clean_data(senti_df)
    senti_df = cleaner.transform_data(senti_df)

    text_preprocess = Text_Preprocessing()
    text_preprocess.preprocess_text(senti_df)
if __name__ == "__main__":
    main()
