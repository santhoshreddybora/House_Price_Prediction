import os
import sys

import numpy as np
import pandas as pd

from HPP.entity.config_entity import HPPredictorConfig
from HPP.entity.s3_estimator import HPPEstimator
from HPP.exception import CustomException
from HPP.logger import logging
from HPP.utils.main_utils import read_yaml
from pandas import DataFrame

class HPPData:
    def __init__(self,
                location,
                no_of_BHK,
                total_sqft,
                bath,
                ):
        """
        Usvisa Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.location = location
            self.no_of_BHK = no_of_BHK
            self.total_sqft = total_sqft
            self.bath = bath
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_hpp_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            usvisa_input_dict = self.get_hpp_data_as_dict()
            return DataFrame(usvisa_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_hpp_data_as_dict(self):
        """
        This function returns a dictionary from USvisaData class input 
        """
        logging.info("Entered get_usvisa_data_as_dict method as USvisaData class")

        try:
            input_data = {
                "location": [self.location],
                "no_of_BHK": [self.no_of_BHK],
                "total_sqft": [self.total_sqft],
                "bath": [self.bath]
              }

            logging.info("Created usvisa data dict")

            logging.info("Exited get_usvisa_data_as_dict method as USvisaData class")

            return input_data

        except Exception as e:
            raise CustomException(e, sys) from e

class HppClassifier:
    def __init__(self,prediction_pipeline_config: HPPredictorConfig = HPPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise CustomException(e, sys)
    def predict(self,dataframe)->str:
        """
        This is the method of USvisaClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of HPP class")
            model = HPPEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise CustomException(e, sys)