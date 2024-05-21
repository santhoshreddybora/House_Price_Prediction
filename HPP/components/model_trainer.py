import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from neuro_mf import ModelFactory

from HPP.exception import CustomException
from HPP.logger import logging

from HPP.utils.main_utils import load_numpy_array_data, read_yaml, load_object, save_object
from HPP.entity.config_entity import ModelTrainerConfig
from HPP.entity.artifact_entity import (DataTransformationArtifact, ModelTrainerArtifact,
                                        RegressionMetricArtifact)
from HPP.entity.estimator import HPPModel



class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.data_transformation_artifact=data_transformation_artifact
        self.model_trainer_config=model_trainer_config
    
    def get_model_object_and_report(self,train:np.array,test:np.array)->Tuple[object,object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Starting model training")
            model_factory=ModelFactory(self.model_trainer_config.model_config_file_path)
            x_train,x_test,y_train,y_test=train[:,:-1],test[:,:-1],train[:,-1],test[:,-1]
            best_model_detail=model_factory.get_best_model(x_train,y_train,
                                                           base_accuracy=self.model_trainer_config.expected_accuracy)
            model_object=best_model_detail.best_model
            y_pred=model_object.predict(x_test)
            r2score=r2_score(y_test,y_pred)
            meanabsoluteerror=mean_absolute_error(y_test,y_pred)
            meansquareerror=mean_squared_error(y_test,y_pred)
            rootmeansquareerror=np.sqrt(meansquareerror)
            metric_artifact=RegressionMetricArtifact(r2score,meanabsoluteerror,meansquareerror,rootmeansquareerror)
            logging.info(f"Got best model object and report")
            return best_model_detail,metric_artifact
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)

    def initiate_model_training(self,)->ModelTrainerArtifact:
        logging.info(f"Enter initiate_model_trainer method of modeltrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            train_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail,metric_artifact=self.get_model_object_and_report(train=train_arr,test=test_arr)
            preprocessing_obj=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            hpp_model=HPPModel(preprocessing_obj=preprocessing_obj,
                              train_model_object=best_model_detail.best_model)
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_file_path, hpp_model)

            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                        metric_artifact=metric_artifact)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)        
