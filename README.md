# House-Price-Prediction


## End to End Workflow

* 1) Template - Creating a template where the project structure is created 
* 2) Setup and Requirements installation
* 3) Logging and Exception and utility files - Log the error or any other info and raise custom expection and also utility files where common used operations where kept that
* 4) EDA and Feature Engineering - Creating a ipynb to get insights from it
* 5) Workflow for all components
     * 1)Constant updation
     * 2)Entity updation
     * 3)Components updation
     * 4)pipeline updation
* 6) App.py for predicting in server 


## Exporting Mongodb url to env variable
```bash
export MONGODB_URL="Connection_url="mongodb+srv://<username>:<pwd>@cluster0.ecs8wjl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
```
## Exporting Aws_access_key and aws_secret_key to env variable
```bash
export AWS_ACCESS_KEY="your key"
export AWS_SECRET_KEY="your secret key"
```

## AWS CICD Deployment with Gitbub actions
1. Login to AWS console
2. Create a IAM user for depolyment

     #with specific access
     1. EC2 access : It is virtual machine

     2. ECR: Elastic Container registry to save your docker image in aws

     #Description: About the deployment

     1. Build docker image of the source code

     2. Push your docker image to ECR

     3. Launch Your EC2 

     4. Pull Your image from ECR in EC2

     5. Lauch your docker image in EC2

     #Policy:

     1. AmazonEC2ContainerRegistryFullAccess

     2. AmazonEC2FullAccess

3. Create a EC2 repo to save/stor docker image
```
- Save the URI: 136566696263.dkr.ecr.us-east-1.amazonaws.com/mlproject
```
4. create a EC2 machine(Ubuntu)

5. Open EC2 and install docker in EC2 machine 

#optinal
```bash
sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```
6. Configure EC2 as a self hosted runner on git hub:
```
setting>actions>runner>new self hosted runner> choose os> then run command one by one
```
7. Git hib secrets
```
* AWS_ACCESS_KEY
* AWS_SECRET__KEY
* AWS_DEFAULT_REGION 
* ECR_REPO - ECR repo name 
```` 