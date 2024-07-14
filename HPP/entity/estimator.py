import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from HPP.exception import CustomException
from HPP.logger import logging
import sys 

class HPPModel:
    def __init__(self, preprocessing_obj: Pipeline, train_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_obj
        self.trained_model_object = train_model_object
    
    def predict(self,dataframe:DataFrame)-> DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of USvisaModel class")
        try:
            logging.info("Using the trained model to get predictions")
            transformed_feature=self.preprocessing_object.transform(dataframe)
            print(transformed_feature.toarray())
            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise CustomException(e,sys)
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"