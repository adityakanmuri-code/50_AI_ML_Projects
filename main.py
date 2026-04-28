import logger

from Customer_Churn_Model_ANN.data_ingestion import DataIngestion
from Customer_Churn_Model_ANN.eda_data_transformation import CleanData,EDA

def main():
    ingestion = DataIngestion()
    df = ingestion.ingest_data()

    #print(df.head())
    #print(df['Geography'].value_counts())
    #print(df['Gender'].value_counts())

    cleaner = CleanData()
    df = cleaner.clean_data(df)
    df = cleaner.transform_data(df)
    eda = EDA(df)
    eda.plot_data()


if __name__ == "__main__":
    main()
