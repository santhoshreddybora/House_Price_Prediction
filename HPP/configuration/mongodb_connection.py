from distutils.sysconfig import customize_compiler
import sys
from HPP.constants import *
from HPP.exception import CustomException
from HPP.logger import logging
import os
import pymongo
import certifi

ca = certifi.where()
##if any timeout issue it will handle

class MongoDBClient:
    """
    Class Name :   export_data_into_feature_store
    Description :   This method exports the dataframe from mongodb feature store as dataframe 
    
    Output      :   connection to mongodb database
    On Failure  :   raises an exception
    """
    client=None
    def __init__(self,database_name=DATABASE_NAME)->None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url=os.getenv(MONGODB_URL)
                if mongo_db_url is None:
                    raise Exception(f"Please provide {MONGODB_URL} environment variable")
                MongoDBClient.client=pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)
                self.client=MongoDBClient.client
                self.database=self.client[database_name]
                self.database_name=database_name
                logging.info("Mongo db connection established")
        except Exception as e:
            raise CustomException(e,sys)
                