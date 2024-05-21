import boto3
import os
from HPP.constants import AWS_SECRET_KEY,AWS_ACCESS_KEY,REGION_NAME

class S3Client:
    s3_client=None
    s3_resource=None
    def __init__(self,region_name=None):
        """ 
        This Class gets aws credentials from env_variable and creates an connection with s3 bucket 
        and raise exception when environment variable is not set
        """
        if S3Client.s3_resource==None and S3Client.s3_client==None:
            __access_key_id=os.getenv(AWS_ACCESS_KEY)
            __secret_access_key=os.getenv(AWS_SECRET_KEY)
            if __access_key_id is None :
                raise Exception(f"AWS_ACCESS_KEY {AWS_ACCESS_KEY} is not set")
            if __secret_access_key is None :
                raise Exception(f"AWS_SECRET_ACCESS_KEY {AWS_SECRET_KEY} is not set")
            

            S3Client.s3_resource=boto3.resource('s3',
                                                aws_access_key_id=__access_key_id,
                                                aws_secret_access_key=__secret_access_key,
                                                region_name=region_name
                                                )
            S3Client.s3_clinet=boto3.client('s3',
                                            aws_access_key_id=__access_key_id,
                                               aws_secret_access_key=__secret_access_key,
                                               region_name=region_name)
            self.s3_resource=S3Client.s3_resource
            self.s3_client = S3Client.s3_client