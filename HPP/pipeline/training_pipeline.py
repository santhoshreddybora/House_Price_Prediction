import os,sys
from HPP.components.data_ingestion import DataIngestion
from HPP.components.data_validation import DataValidation
from HPP.components.data_transformation import DataTransformation
from HPP.components.model_trainer import ModelTrainer
from HPP.components.model_evaluation import ModelEvaluation
from HPP.components.model_pusher import ModelPusher
from HPP.entity.config_entity import (DataIngestionConfig,DataValidationConfig,
                                      DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig,
                                      ModelPusherConfig)
from HPP.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact,
                                        DataTransformationArtifact,ModelTrainerArtifact,
                                        ModelEvaluationArtifact,ModelPusherArtifact)
from HPP.logger import logging
from HPP.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()
        self.data_transformation_config=DataTransformationConfig()
        self.model_trainer_config=ModelTrainerConfig()
        self.model_evaluation_config=ModelEvaluationConfig()
        self.model_pusher_config=ModelPusherConfig()

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
        
    
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            logging.info("Starting data validation method of TrainingPipeline class")
            data_validation=DataValidation(self.data_ingestion_config,self.data_validation_config)
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info(f"Performed the data validation operation")
            logging.info(f"Exited start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)

    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact,
                                  data_validation_artifact:DataValidationArtifact,
                                  data_transformation_config=DataTransformationConfig)->DataTransformationArtifact:
        try:
            logging.info("Starting data transformation method of TrainingPipeline class")
            data_transformation=DataTransformation(data_ingestion_artifact=data_ingestion_artifact
                                                ,data_validation_artifact=data_validation_artifact,
                                                data_transformation_config=data_transformation_config)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
    def model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for training the model
        """
        try:
            logging.info("Entered the train_pipeline method of TrainPipeline class")
            model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                       model_trainer_config=self.model_trainer_config)
            model_trainer_artifact=model_trainer.initiate_model_training()
            logging.info("Exited the train_pipeline method of TrainPipeline class")
            return model_trainer_artifact
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               model_trainer_artifact: ModelTrainerArtifact,
                               data_transformation_artifact:DataTransformationArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting modle evaluation
        """
        try:
            model_evaluation = ModelEvaluation(model_eval_config=self.model_evaluation_config,
                                               data_ingestion_artifact=data_ingestion_artifact,
                                               model_trainer_artifact=model_trainer_artifact,
                                               data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            model_pusher=ModelPusher(model_pusher_config=self.model_pusher_config,
                                     model_evaluation_artifact=model_evaluation_artifact)
            model_pusher_artifact=model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    def run_pipeline(self):
        """
        This method of TrainPipeline class is responsible for running the pipeline
        """
        try:
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_ingestion_artifact,data_validation_artifact)
            model_trainer_artifact=self.model_trainer(data_transformation_artifact)
            model_evaluation_artifact=self.start_model_evaluation(data_ingestion_artifact,model_trainer_artifact,data_transformation_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                logging.info(f"Model is not accepted")
                raise CustomException(e,sys)
            model_pusher_artifact=self.start_model_pusher(model_evaluation_artifact)
            logging.info("Exited the run_pipeline method of TrainPipeline class")
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
6