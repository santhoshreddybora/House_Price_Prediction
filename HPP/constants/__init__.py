import os

"""Mongo db connection constants"""
DATABASE_NAME="House_Price_Prediction"
COLLECTION_NAME="HPP"
MONGODB_URL="MONGODB_URL"


PIPELINE_NAME:str="HPP"
ARTIFACT_DIR:str="artifact"

FILE_NAME:str='HPP.csv'
TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"
SCHEMA_FILE_PATH:str=os.path.join('config','schema.yaml')



"""
Data Ingestion related constants
"""
DATA_INGESTION_COLLECTION_NAME:str="HPP"
DATA_INGESTION_DIR_NAME:str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str="feature_store"
DATA_INGESTION_INGESTED_DIR:str="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2

"""
Data validation related constants
"""
DATA_VALIDATION_DIR_NAME:str="validation"
DATA_VALIDATION_DRIFT_REPORT_DIR:str="drift_report"
DATA_VALIDATION_REPORT_FILE_NAME:str="report.yaml"



"""
Data transformation related constants
"""

DATA_TRANSFORMATION_DIR_NAME:str="transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str="transformed"
DATA_TRANSFORMATION_OBJECT_DIR:str="transformed_object"
PREPROCESSING_OBJECT_FILE_NAME="preprocessing.pkl"


