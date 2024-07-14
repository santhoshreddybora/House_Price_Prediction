from HPP.entity.config_entity import ModelEvaluationConfig
from HPP.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact,DataTransformationArtifact
from sklearn.metrics import r2_score
from HPP.exception import CustomException
from HPP.constants import TARGET_COLUMN
from HPP.logger import logging
import sys
import pandas as pd
from typing import Optional
from HPP.entity.s3_estimator import HPPEstimator
from dataclasses import dataclass
from HPP.entity.estimator import HPPModel
from pathlib import Path
from HPP.constants import SCHEMA_FILE_PATH,TARGET_COLUMN
from HPP.utils.main_utils import (save_object, save_numpy_array_data, read_yaml,
                                   drop_columns,convert_sqft_to_num,remove_pps_outliers,
                                   remove_bhk_outliers)

@dataclass
class EvaluateModelResponse:
    trained_model_r2_score:float
    best_model_r2_score:float
    is_model_accepted:bool
    difference:float

class ModelEvaluation:
    def __init__(self,model_eval_config:ModelEvaluationConfig,data_ingestion_artifact:DataIngestionArtifact,
                 model_trainer_artifact:ModelTrainerArtifact,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self._schema_config =read_yaml(filepath=SCHEMA_FILE_PATH)
        except Exception as e:
            CustomException(e,sys)
    
    def get_best_model(self) -> Optional[HPPEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            Hpp_estimator = HPPEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if Hpp_estimator.is_model_present(model_path=model_path):
                return Hpp_estimator
            return None
        except Exception as e:
            raise  CustomException(e,sys)
    
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
                        

            drop_cols = self._schema_config['drop_columns']
            print(drop_cols)
            df1=drop_columns(df=test_df,cols=drop_cols)
            df1=df1.dropna()
            df1['no_of_BHK']=df1['size'].apply(lambda x: int(x.split(' ')[0]) if pd.notna(x) else None)
            df1['total_sqft']=df1['total_sqft'].apply(convert_sqft_to_num)
            df1['price_per_sqft']=df1['price']*100000/(df1['total_sqft'])
            location_count=df1['location'].value_counts()
            location_less_than_10=location_count[location_count<=10]
            df1['location']=df1['location'].apply(lambda x:"other" if x in location_less_than_10 else x)
            df2=df1[~(df1['total_sqft']/df1['no_of_BHK']<300)]
            df3=remove_pps_outliers(df2)
            df4=remove_bhk_outliers(df3)
            df5=df4[df4.bath<df4.no_of_BHK+2]
            df6=drop_columns(df=df5,cols=['size','price_per_sqft'])
            test_df=df6
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_r2_score = self.model_trainer_artifact.metric_artifact.r2score

            best_model_r2_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_r2_score = r2_score(y, y_hat_best_model)
            
            tmp_best_model_score = 0 if best_model_r2_score is None else best_model_r2_score
            result = EvaluateModelResponse(trained_model_r2_score=trained_model_r2_score,
                                           best_model_r2_score=best_model_r2_score,
                                           is_model_accepted=trained_model_r2_score > tmp_best_model_score,
                                           difference=trained_model_r2_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e