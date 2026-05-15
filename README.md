# 🚀 50 End-to-End Machine Learning Projects

Welcome to the **50 End-to-End Machine Learning Projects Repository**.

This repository is a collection of practical, production-oriented Machine Learning and Deep Learning projects designed to cover:

- Data Ingestion
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning
- Deep Learning
- Hyperparameter Tuning
- NLP
- MLOps
- Deployment
- Prediction Pipelines
- Config-Driven ML Architectures

The goal of this repository is to build industry-level Machine Learning projects with:
- Modular Coding Standards
- Scalable Architecture
- Reusable Components
- Production-Ready Pipelines
- Logging & Exception Handling
- Configuration-Based Workflow

---

# 📌 Project 1: Customer Churn Prediction using ANN

The first project in this repository focuses on predicting customer churn using an **Artificial Neural Network (ANN)**.

This project demonstrates:
- End-to-End ML Pipeline
- Config-Driven Data Processing
- ANN Model Training
- Hyperparameter Tuning using Keras Tuner
- Modular Data Transformation
- Prediction on Unseen Data
- Reusable Transformer Pipelines

---

# 🎯 Problem Statement

Customer churn prediction helps businesses identify customers who are likely to discontinue their services.

Using customer demographic and behavioral data, the ANN model predicts whether a customer is likely to churn or continue using the service.

---

# 📂 Dataset Information

## 📘 Training Dataset

The training dataset used in this project was taken from the following course:

Krish Naik Machine Learning Course:

https://www.udemy.com/course/complete-machine-learning-nlp-bootcamp-mlops-deployment/learn/lecture/44526479#overview

---

## 📘 Test Dataset

The test dataset was synthetically generated using ChatGPT.

### Prompt Used for Dataset Generation

```text
Generate the dataset in downloadable csv using the attached csv file as reference.
Maintain the same schema as in the attached csv sheet.

Folder Structure for Customer Churn Model ANN
Customer_Churn_Model_ANN/
│
├── artifacts/
│   ├── models/
│   │   ├── dl_model.h5
│   │   └── preprocessor.pkl
│   │
│   ├── plots/
│   │
│   └── logs/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── train/
│   │   └── churn_train.csv
│   │
│   └── predict/
│       └── churn_predict.csv
│
├── notebooks/
│
├── Customer_Churn_Model_ANN/
│   ├── data_ingestion.py
│   ├── eda_data_transformation.py
│   ├── transformer_factory.py
│   ├── model_training.py
│   ├── model_prediction.py
│   ├── configuration.py
│   ├── logger.py
│   ├── exception.py
│   └── __init__.py
│
├── hyperparameters/
│
├── requirements.txt
│
├── main.py
│
└── README.md
⚙️ Config-Driven Architecture

The entire pipeline is controlled using config.yaml.

All major parameters are dynamically fetched from the configuration file, including:

Dataset paths
Source type
Read parameters
Cleaning steps
Transformation steps
Feature engineering configuration
ANN architecture
Hyperparameters
Optimizer settings
Loss functions
Metrics
Callbacks
Output directories
Logging paths

This architecture provides:

Better scalability
Easier experimentation
Reusable workflows
Improved maintainability
📥 Data Ingestion

The DataIngestion module dynamically loads datasets based on the configuration settings.

Features
Supports CSV files
Supports Excel files
Config-driven file reading
Dynamic read parameters
Logging support
Exception handling
Supported Source Types
File-based ingestion
🧹 Data Cleaning & Transformation

The CleanData class performs all preprocessing operations dynamically using configurable transformation pipelines.

Supported Cleaning Operations
Drop unwanted columns
Handle missing values
Strip whitespaces
Convert datatypes
Encode categorical variables
Fix outliers
Clean column names
Outlier Handling Techniques
IQR Method

Outliers are detected using:

IQR=Q
3
	​

−Q
1
	​


Z-Score Method

Outliers are also supported using the Z-score technique.

Supported Outlier Actions
Clip outliers
Drop outlier rows
📊 Exploratory Data Analysis (EDA)

The EDA module automatically generates visualizations based on the configuration settings.

Supported Visualizations
Numerical Distribution Plots
Categorical Distribution Plots
Correlation Heatmaps
Target Relationship Analysis
Outlier Plots
Plot Storage

Generated plots are automatically stored in timestamp-based folders.

🔄 Transformer Factory

The Transformer_Factory dynamically creates preprocessing transformers.

Supported Transformers
StandardScaler
OneHotEncoder
OrdinalEncoder
TargetEncoder

This enables:

Flexible preprocessing pipelines
Reusable transformer logic
Configurable preprocessing architecture
🧠 ANN Model Training

The Model_Trainer class handles complete ANN model training.

🔥 Hyperparameter Tuning

Hyperparameter tuning is implemented using Keras Tuner RandomSearch.

Tuned Parameters
Number of Hidden Layers
Number of Neurons
Learning Rate
ANN Workflow
Input Layer
   ↓
Hidden Layers
   ↓
Output Layer

The ANN architecture is dynamically built using values from config.yaml.

⚡ Optimizers Supported
Adam
SGD
RMSProp
📉 Supported Loss Functions
Binary Crossentropy
Categorical Crossentropy
Mean Squared Error
Huber Loss
🔔 Callbacks Supported
EarlyStopping
TensorBoard

These callbacks help:

Prevent overfitting
Improve training efficiency
Monitor model performance
💾 Model Persistence

The following artifacts are saved after training:

Artifact	Description
preprocessor.pkl	Saved preprocessing pipeline
dl_model.h5	Trained ANN model

This enables seamless prediction on unseen datasets.

🔮 Customer Churn Prediction Usage

A separate prediction pipeline is implemented using the Prediction class.

▶️ Prediction Workflow

The prediction pipeline performs the following operations:

Load unseen customer dataset
Apply cleaning pipeline
Apply transformation pipeline
Load saved preprocessing object
Load trained ANN model
Transform unseen data
Predict customer churn
Calculate prediction accuracy
▶️ How to Run Customer Churn Prediction
Step 1: Configure Prediction Dataset

Update the prediction dataset path in config.yaml.

Example:

data:
  predict:
    source_type: file
    file_path: data/predict/customer_churn_predict.csv
Step 2: Run the Pipeline
python main.py
Step 3: Output

The model will:

Load the trained ANN model
Load preprocessing pipeline
Predict customer churn
Generate prediction accuracy
📈 Technologies Used
Machine Learning & Deep Learning
TensorFlow
Keras
Keras Tuner
Scikit-learn
Data Processing
Pandas
NumPy
Visualization
Matplotlib
Seaborn
📌 Key Learnings

This project demonstrates:

End-to-End ML Workflow
Config-Driven Pipelines
ANN Model Development
Hyperparameter Tuning
Production-Level Project Structure
Modular Code Design
Logging & Exception Handling
Model Serialization
Prediction Pipeline