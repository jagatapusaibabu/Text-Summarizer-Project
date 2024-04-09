# Text-Summarizer-Project

## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. update the conponents
6. update the pipeline
7. update the main.py
8. update the app.py
9. Run the application using `python -m text_summarization` or `flask'

streamlit run "C:\Users\jagat\Desktop\4th-year-project\Text-Summarizer-Project\app.py"

  <!-- You can now view your Streamlit app in your browser. -->
  <!-- #Local URL: http://localhost:8501 -->
  <!-- #Network URL: http://192.168.218.64:8501 -->


Note: The project is built on Flask

The project uses a pretrained model (bert-base-uncased) for tokenizing and generating summaries, which can be found here
""" Configurations"""
The configurations are divided into two files,
config.yaml : Contains general information about the
application and its dependencies. It includes details like the name of the model, the preprocessing methods to be used etc.
This file should not be updated frequently as it contains static information.

params.yaml : This file contains dynamic parameters which can change from run to run.It may include things like the maximum length of
summary sentences, the number of summary sentences, the method to use for summarizing etc.
These values can be changed according to the requirements of different projects.
params.yaml : Contains dynamic parameters that
"""Entities"""
There are three types of entities that you need to define in order to make your code work properly. Here are some examples
There are three types of entities that need to be defined in order to make the code work properly
1.Tokenizer : A class representing a tokenizer object. It will be used by the Preprocessor component to convert raw text into tokens.. They are:

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/entbappy/End-to-end-Text-Summarization
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n summary python=3.8 -y
```

```bash
conda activate summary
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


```bash
Author: Krish Naik
Data Scientist
Email: krishnaik06@gmail.com

```



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

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

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/text-s

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app


	run command :streamlit run "C:\Users\jagat\Downloads\trial-2\Text-Summarizer-Project\app.py"
	