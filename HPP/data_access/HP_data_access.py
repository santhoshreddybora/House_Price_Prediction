from HPP.configuration.mongodb_connection import MongoDBClient
from HPP.constants import DATABASE_NAME
from HPP.exception import CustomException
from typing import Optional
import pandas as pd
import numpy as np
import sys


class HPData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """
    def __init__(self,):
        try:
            self.mongo_client=MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e,sys)
    def export_collection_as_df(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            """export entire collectin as dataframe:
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection=self.mongo_client.database[collection_name]
            else:
                collection=self.mongo_client[database_name][collection_name]
            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(["_id"],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise CustomException(e,sys)