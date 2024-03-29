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

