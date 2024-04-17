### Clasiffication-DeepLearning-Practice

### Work-Flow

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py


## ML Flow Credentials

export MLFLOW_TRACKING_URI = https://dagshub.com/shahid-abdul/Clasiffication-DeepLearning-Practice.mlflow \
export MLFLOW_TRACKING_USERNAME = shahid-abdul \
export MLFLOW_TRACKING_PASSWORD = 4976d418dad68a72609014529cbdb9c6712dc2e4 \


## dvc commands

These are the dvc commands

1. dvc init
2. dvc repro
3. dvc dag

## Login to your AWS console

## aws deployment stage

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

## Create a ECR repo

## Create ec2 machine Ubuntu

## Open EC2 and Install docker in EC2 Machine:

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

## Configure EC2 as self-hosted runner:

## Setup github secrets:

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = 

AWS_ECR_LOGIN_URI = 

ECR_REPOSITORY_NAME = s