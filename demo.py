from HPP.exception import CustomException
from HPP.pipeline.training_pipeline import TrainingPipeline
import sys
try:
    obj=TrainingPipeline()
    obj.run_pipeline()
except Exception as e:
    raise CustomException(e,sys)