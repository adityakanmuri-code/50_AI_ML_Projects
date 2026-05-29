
from Amazon_Reviews_Sentimment_Analysis.data_ingestion import DataIngestion
from Amazon_Reviews_Sentimment_Analysis.eda_data_transformation import CleanData
from Amazon_Reviews_Sentimment_Analysis.text_preprocessing import Text_Preprocessing
from Amazon_Reviews_Sentimment_Analysis.model_training import Model_Trainer


def main():
    ingestion = DataIngestion(data_type='train')
    senti_df = ingestion.ingest_data()

    cleaner = CleanData()
    senti_df = cleaner.clean_data(senti_df)
    senti_df = cleaner.transform_data(senti_df)

    text_preprocess = Text_Preprocessing()
    senti_df = text_preprocess.preprocess_text(senti_df)

    trainer = Model_Trainer(senti_df)
    trainer.model_trainer()
if __name__ == "__main__":
    main()
