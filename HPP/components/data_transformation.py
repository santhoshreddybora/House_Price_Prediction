import os
from HPP.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact,DataValidationArtifact
from HPP.entity.config_entity import DataTransformationConfig
from HPP.exception import CustomException
from HPP.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from HPP.utils.main_utils import (save_object, save_numpy_array_data, read_yaml,
                                   drop_columns,convert_sqft_to_num,remove_pps_outliers,
                                   remove_bhk_outliers)
from HPP.constants import SCHEMA_FILE_PATH,TARGET_COLUMN



##This class is responsible for fit and transform the data into desired format we can add any conditions here
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            def internal(df):
                # Your feature engineering steps here
                drop_cols = ['area_type','availability','society','balcony']
                df1=drop_columns(df,cols=drop_cols)
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
                df6=df5.drop(['size','price_per_sqft'],axis=1)
                if df6.isna().sum().sum() > 0:
                    raise ValueError("NaN values detected before encoding")
                # encoder = OneHotEncoder(sparse_output=False)
                X_encoded = pd.get_dummies(df6.location,dtype='int')  # Assuming 'location' is the column to be encoded
                # Convert encoded features to DataFrame
                # df_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['location']))               # Concatenate encoded features with other features
                df_final = pd.concat([df6.drop(columns=['location']), X_encoded.drop(['other'],axis=1)], axis=1)
                return df_final
            preprocessed_data = internal(X)
            return preprocessed_data
        except Exception as e:
            raise CustomException(e, sys) from e







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
            feature_engineer = FeatureEngineer()

            # preprocessor = ColumnTransformer(
            # transformers=[
            #         ('passthrough', 'passthrough', ['no_of_BHK'])  # Add your encoded columns here
            #         ],remainder='passthrough'  # Keep the remaining columns unchanged
            #         )

            # Combine FeatureEngineer and preprocessor into a Pipeline
            pipeline = Pipeline([
            ('feature_engineer', feature_engineer)
            # ('preprocessor', preprocessor)
                ])

            logging.info(f"Created Preprocessor object from columntransformer")

            return pipeline
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
                Hpp_df=DataTransformation.read_data(self.data_ingestion_artifact.feature_store_path)
                # test_df=DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)

                transformed_df=preprocessor.fit_transform(Hpp_df)
                logging.info("drop the columns in drop_cols of Train dataset")
                print(transformed_df)
                X=drop_columns(transformed_df,cols=['price'])
                y=transformed_df['price']

                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
                train_arr=np.concatenate([np.array(X_train),np.array(y_train).reshape(-1,1)],axis=1)
                test_arr=np.concatenate([np.array(X_test),np.array(y_test).reshape(-1,1)],axis=1)   
                print(train_arr.shape)
                print(test_arr.shape)         
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
    
        
