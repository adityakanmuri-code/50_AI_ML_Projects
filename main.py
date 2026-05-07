import logger

from Customer_Churn_Model_ANN.data_ingestion import DataIngestion
from Customer_Churn_Model_ANN.eda_data_transformation import CleanData,EDA
from Customer_Churn_Model_ANN.model_training import Model_Trainer
from Customer_Churn_Model_ANN.model_prediction import Prediction

def main():
    ingestion = DataIngestion(data_type='train')
    df = ingestion.ingest_data()

    #print(df.head())
    #print(df['Geography'].value_counts())
    #print(df['Gender'].value_counts())

    cleaner = CleanData()
    df = cleaner.clean_data(df)
    df = cleaner.transform_data(df)
    #eda = EDA(df)
    #eda.plot_data()

    trainer = Model_Trainer(df)
    trainer.model_trainer()

    predict = Prediction()
    predict.predict_data()

if __name__ == "__main__":
    main()
