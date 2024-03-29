import json
import sys
import pandas as pd

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from HPP.exception import CustomException
from HPP.logger import logging
from HPP.utils.main_utils import read_yaml,write_yaml

from HPP.entity.config_entity import DataValidationConfig
from HPP.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from HPP.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config=read_yaml(filepath=SCHEMA_FILE_PATH)
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status=len(dataframe.columns)==len(self._schema_config["columns"])
            logging.info(f"Is required columns present:[{status}]")
            return status
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
    def is_columns_exists(self,dataframe:pd.DataFrame)->bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns=dataframe.columns
            missing_numerical_columns=[]
            missing_categorical_columns=[]
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            if len(missing_categorical_columns)>0:
                logging.info(f"Missing numerical column: {missing_categorical_columns}")
            return False if len(missing_numerical_columns)>0 or len(missing_categorical_columns)>0 else True
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
    
    def detect_data_drift(self,reference_df:pd.DataFrame,current_df:pd.DataFrame)->bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method validates if drift is detected
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            data_drift_profile=Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference_data=reference_df,current_data=current_df)
            report=data_drift_profile.json()
            json_report=json.loads(report)
            write_yaml(filepath=self.data_validation_config.drift_report_file_path,content=json_report)
            n_features=json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features=json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]
            logging.info(f"{n_drifted_features}/{n_features} drift detected")
            drift_status=json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def initiate_data_validation(self,)->DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            validation_error_msg=""
            logging.info("starting data validation")
            train_df,test_df=(DataValidation.read_data(file_path=self.data_ingestion_artifact.train_file_path),
                              DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            status=self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training data frame: {status}")
            if not status:
                validation_error_msg+=f"Columns are missing in training data frame"
            
            status=self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing data frame: {status}")
            if not status:
                validation_error_msg+=f"Columns are missing in testing data frame"
            
            status=self.is_columns_exists(dataframe=train_df)
            if not status:
                validation_error_msg+=f"Numerical or Categorical Columns are missing in training data frame"
            
            status=self.is_columns_exists(dataframe=test_df)
            if not status:
                validation_error_msg+=f"Numerical or Categorical Columns are missing in testing data frame"
            validation_status=len(validation_error_msg)==0
            if validation_status:
                drift_status=self.detect_data_drift(train_df,test_df)
                if drift_status:
                    logging.info(f"Drift Detected")
                    validation_error_msg="Drift Detected"
                else:
                    validation_error_msg="Drift Not Detected"
            logging.info(f"Validation_error:{validation_error_msg}")

            data_validation_artifact=DataValidationArtifact(validation_status=validation_status,
                                                            message=validation_error_msg,
                                                            drift_report_file_path=self.data_validation_config.drift_report_file_path)

            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact        
        except Exception as e:
            raise CustomException(e,sys)
            