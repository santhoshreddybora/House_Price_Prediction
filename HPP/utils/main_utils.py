import os 
import sys

import numpy as np 
import dill 
import yaml

import pandas as pd
from HPP.exception import CustomException
from HPP.logger import logging

def read_yaml(filepath:str)->dict:
    try:
        with open(filepath,"rb") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys)
    

def write_yaml(filepath:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(filepath):
                os.remove(filepath)
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        with open(filepath,"w") as f:
            yaml.dump(content,f)
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys)

def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")
        return obj

    except Exception as e:
        raise CustomException(e, sys) from e
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e
    

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_object(filepath:str,obj:object)->None:
    logging.info("Entered the save obeject method in utils file")

    try :
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath,'wb') as f:
            dill.dump(obj,f)
        logging.info("dumped the object into filepath ")
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys) from e

def drop_columns(df: pd.DataFrame, cols: list)-> pd.DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise CustomException(e, sys) from e