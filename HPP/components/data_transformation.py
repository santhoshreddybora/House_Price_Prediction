import os
from HPP.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact,DataValidationArtifact
from HPP.entity.config_entity import DataTransformationConfig
from HPP.exception import CustomException
from HPP.logger import logging

import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

from HPP.utils.main_utils import (save_object, save_numpy_array_data, read_yaml,
                                   drop_columns,convert_sqft_to_num,remove_pps_outliers,
                                   remove_bhk_outliers)
from HPP.constants import SCHEMA_FILE_PATH,TARGET_COLUMN


class DataTransformation:
    def __init__(self,data_ingestion_artifact=DataIngestionArtifact,
                 data_validation_artifact=DataValidationArtifact,
                 data_transformation_config=DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config =read_yaml(filepath=SCHEMA_FILE_PATH)
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
    def get_data_transformer_object(self)->Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )
        try:
            logging.info("Got Numerical columns from schema config")
            numeric_transformer=StandardScaler()
            logging.info("Initialize StandardScaler")

            num_features=self._schema_config['num_features']

            preprocessor=ColumnTransformer([
                ('StandardScaler',numeric_transformer,num_features)
            ])

            logging.info(f"Created Preprocessor object from columntransformer")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    @staticmethod
    def feature_engineering(df1)->pd.DataFrame:
        try:
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
            df6=df5.drop(['size','price_per_sqft'],axis=1)
            dummies=pd.get_dummies(df6.location,dtype='int')
            df6=pd.concat([df6,dummies.drop('other',axis='columns')],axis='columns')
            df7=df6.drop('location',axis='columns')
            logging.info(df7.columns)
            return df7
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys) from e
        
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info(f"Starting data transformation")
                preprocessor=self.get_data_transformer_object()
                logging.info(f"Got preprocessor object")
                train_df=DataTransformation.read_data(self.data_ingestion_artifact.train_file_path)
                test_df=DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)


                logging.info("drop the columns in drop_cols of Train dataset")
                drop_cols = self._schema_config['drop_columns']
                df1=train_df.drop(columns=drop_cols,axis=1)
                df1=DataTransformation.feature_engineering(df1)
                input_feature_train_df=df1.drop(columns=[TARGET_COLUMN],axis=1)
                output_feature_train_df=df1[TARGET_COLUMN]
                logging.info("Got train features of Training dataset")

                

                logging.info("drop the columns in drop_cols of Test dataset")
                df2=test_df.drop(columns=drop_cols,axis=1)
                df2=DataTransformation.feature_engineering(df2)
                input_feature_test_df=df2.drop(columns=[TARGET_COLUMN],axis=1)
                output_feature_test_df=df2[TARGET_COLUMN]

                logging.info("Got test features of Testing dataset")

                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )


                input_feature_train_arr=np.array(input_feature_train_df)
                logging.info("Used the preprocessor object to transform the test features")

                input_feature_test_arr=np.array(input_feature_test_df)
                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Created train array and test array")
                train_arr=np.c_[input_feature_train_arr,np.array(output_feature_train_df)]
                test_arr=np.c_[input_feature_test_arr,np.array(output_feature_test_df)]
                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )
                data_transformation_artifact=DataTransformationArtifact(transformed_object_file_path=
                                                                        self.data_transformation_config.transformed_object_file_path,
                                                                        transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                                                                        transformed_test_file_path=self.data_transformation_config.transformed_test_file_path)
                return data_transformation_artifact
            else:
                logging.info(self.data_validation_artifact.message)
                raise Exception(self.data_validation_artifact.message)
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
        
