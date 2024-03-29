import os,sys
from HPP.components.data_ingestion import DataIngestion
from HPP.components.data_validation import DataValidation
from HPP.entity.config_entity import DataIngestionConfig,DataValidationConfig
from HPP.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataValidationArtifact
from HPP.logger import logging
from HPP.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()

    def start_data_ingestion(self)->DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component

        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
        
    
    def start_data_validation(self)->DataValidationArtifact:
        logging.info("Starting data validation method of TrainingPipeline class")
        data_validation=DataValidation(self.data_ingestion_config,self.data_validation_config)
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info(f"Performed the data validation operation")
        logging.info(f"Exited start_data_validation method of TrainPipeline class")
        return data_validation_artifact
    
    
    def run_pipeline(self):
        """
        This method of TrainPipeline class is responsible for running the pipeline
        """
        try:
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation()
            logging.info("Exited the run_pipeline method of TrainPipeline class")
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
        