from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str
    feature_store_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool 
    message:str
    drift_report_file_path:str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class RegressionMetricArtifact:
    r2score:float
    meanabsoluteerror:float
    meansquarederror:float
    rootmeansquareerror:float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:RegressionMetricArtifact