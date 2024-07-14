import os
from HPP.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact,DataValidationArtifact
from HPP.entity.config_entity import DataTransformationConfig,DataIngestionConfig
from HPP.exception import CustomException
from HPP.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from HPP.utils.main_utils import (save_object, save_numpy_array_data, read_yaml,
                                   drop_columns,convert_sqft_to_num,remove_pps_outliers,
                                   remove_bhk_outliers)
from HPP.constants import SCHEMA_FILE_PATH,TARGET_COLUMN
from pathlib import Path



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
            oh_transformer = OneHotEncoder(handle_unknown='ignore')
            st_transformer = StandardScaler()
            oh_columns=self._schema_config['oh_columns']
            num_columns=self._schema_config['num_features']
            
            preprocessor=ColumnTransformer([
                ('OnehotEncoder',oh_transformer,oh_columns),
                ('StandardScaler',st_transformer,num_columns)
               ])
            

            logging.info(f"Created Preprocessor object from columntransformer")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    
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
                # train_df=DataTransformation.read_data(self.data_ingestion_artifact.train_file_path)
                # test_df=DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)
                Total_df=DataTransformation.read_data(self.data_ingestion_artifact.feature_store_path)
                print(Total_df.columns)
                # df=DataTransformation.read_data(self.data_ingestion_artifact.feature_store_path)
                drop_cols = self._schema_config['drop_columns']
                print(drop_cols)
                df1=drop_columns(df=Total_df,cols=drop_cols)
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
                print(df6)
                train_set,test_set=train_test_split(df6,test_size=DataIngestionConfig.train_test_split_ratio)
                # if df6.isna().sum().sum() > 0:
                #     raise ValueError("NaN values detected before encoding")
                logging.info(train_set)
                logging.info(test_set)
                logging.info("drop the columns in drop_cols of Train dataset")
                input_feature_train_df=drop_columns(df=train_set,cols=['price'])
                output_feature_train_df=train_set['price']
                # test_df=DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)
                transformed_train_array=preprocessor.fit_transform(input_feature_train_df)
                print(transformed_train_array.toarray())
                logging.info("drop the columns in drop_cols of Test dataset")
                input_feature_test_df=drop_columns(df=test_set,cols=['price'])
                output_feature_test_df=test_set['price']
                transformed_test_array=preprocessor.transform(input_feature_test_df)
                
                
                transformed_train_array_dense = transformed_train_array.toarray()
                transformed_test_array_dense = transformed_test_array.toarray()

                # Reshape the target arrays if needed
                output_feature_train_df = np.array(output_feature_train_df).reshape(-1, 1)
                output_feature_test_df = np.array(output_feature_test_df).reshape(-1, 1)

                print("Reshaped output feature train array shape:", output_feature_train_df.shape)
                print("Reshaped output feature test array shape:", output_feature_test_df.shape)

                print("Transformed train array (dense) shape:", transformed_train_array_dense.shape)
                print("Output feature train array shape:", output_feature_train_df.shape)

                print("Transformed test array (dense) shape:", transformed_test_array_dense.shape)
                print("Output feature test array shape:", output_feature_test_df.shape)

                # Check dimensions before concatenation
                assert transformed_train_array_dense.shape[0] == output_feature_train_df.shape[0], "Mismatch in number of rows between transformed train array and output feature train array"
                assert transformed_test_array_dense.shape[0] == output_feature_test_df.shape[0], "Mismatch in number of rows between transformed test array and output feature test array"

                # Concatenate transformed arrays with target arrays
                train_arr = np.c_[transformed_train_array_dense, output_feature_train_df]
                test_arr = np.c_[transformed_test_array_dense, output_feature_test_df]

                print("Train array shape after concatenation:", train_arr.shape)
                print("Test array shape after concatenation:", test_arr.shape)

  
                # print(train_arr.shape)
                # print(test_arr.shape)         
                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")
                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )
                data_transformation_artifact=DataTransformationArtifact(transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                                                                        transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                                                                        transformed_test_file_path=self.data_transformation_config.transformed_test_file_path) 
                return data_transformation_artifact
            else:
                logging.info(self.data_validation_artifact.message)
                raise Exception(self.data_validation_artifact.message)
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
        
