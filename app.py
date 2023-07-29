from fastapi import FastAPI
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import time
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base")

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embedding():
  text = '''Let's learn with Ahmad Rosid Blog Subscribe How to Deploy FastApi to Fly.io? FastAPI is a modern, fast (high-performance), web framework for building APIs with Python language. It is a user-friendly, modular, and easy-to-extend framework that makes it easy to get started building APIs with minimal setup and configuration. In this article, you will learn how to deploy a FastAPI application to Fly.io, a cloud platform for hosting and managing modern web applications. Fly.io offers a highly scalable, secure, and performant platform for hosting web applications, making it a great choice for deploying FastAPI applications. Additionally, it offers a free plan, making it an ideal option for you who are in the early stages of building a prototype for their application. We will go through the process of setting up a FastAPI application, configuring it for deployment to Fly.io, and deploying the application to the platform. By the end of this article, you will have a fully functional FastAPI application running on Fly.io. Create FastApi project To create a FastAPI project, you will first need to make sure you have Python 3.7 or higher installed on your system. You can check your Python version by running the following command in a terminal: python --version Once you have Python installed, you can use the pip package manager to install FastAPI and its dependencies. Run the following command to install FastAPI: pip install fastapi With FastAPI installed, you can now create a new FastAPI project. This time we will creating the FastApi project from scratch. To create a new FastAPI project from scratch, create a new directory for your project and create a main.py file inside it. This will be the entry point for your FastAPI application. You can then import FastAPI and create a new FastAPI instance as follows: from fastapi import FastAPI app = FastAPI() You can now define your API endpoints like this: @app.get("/") def read_root(): return {"Hello": "World"} Add healthcheck In order to be able to deploy FastApi project to Fly.io we need to provide healthcheck endpoint. To add a healthcheck endpoint to your FastAPI application, you can use the @app.get decorator to define a function that will be called when the endpoint is accessed. For example: @app.get("/healthcheck") def read_root(): return {"status": "ok"} This will create an endpoint at /healthcheck that returns a JSON object with a status field set to ok. You can then use this endpoint to monitor the health of your application by making a request to it periodically. Add requirement.txt file In order to deploy a FastAPI application to Fly.io, you will need to create a requirement.txt file that lists all of the dependencies required by your application. This file is used by Fly.io to install the necessary packages and libraries when deploying your application. For FastApi project this is the most important pip package you need to provide. fastapi uvicorn It is important to make sure that the requirement.txt file is up to date and includes all of the necessary dependencies for your application. This will ensure that your application can be deployed and run smoothly on Fly.io. You don't want to manually generate the requirements.txt you can use the pip freeze command to generate a list of installed packages and their versions. pip freeze > requirements.txt One thing you need to keep in mind is to include uvicorn package, this package will be used to run the server later in Fly.io. Enable Cors While this is optional, if you plan on consuming the API from a different domain, you may wish to consider enabling CORS in your FastAPI project in order to facilitate cross-origin requests. To enable CORS for a FastAPI project, you can use the FastAPI-CORS library, which provides an easy way to add CORS support to a FastAPI application. To install FastAPI-CORS, run the following command: pip install fastapi-cors Please don't forget to also update your requirements.txt file with this new pip package. fastapi fastapi-cors uvicorn With FastAPI-CORS installed, you can then use the @app.middleware("http") decorator to add a CORS middleware to your application. For example: from fastapi_cors import CORS app = FastAPI() origins = ["*"] app.add_middleware( CORS, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"], ) You also can define allowed domain to acces your API like this. origins = [ "https://example.com", "http://subdomain.example.com", "http://localhost", "http://localhost:3000", ] Add Dockerfile Fly.io supports deployment using a Dockerfile, which is useful for configuring custom dependencies for your application. The following is a basic Dockerfile that can be used for a FastAPI project: # https://hub.docker.com/_/python FROM python:3.10-slim-bullseye ENV PYTHONUNBUFFERED True ENV APP_HOME /app WORKDIR $APP_HOME COPY . ./ RUN pip install --no-cache-dir -r requirements.txt CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] You are welcome to modify the Dockerfile to meet the specific requirements of your application. Please feel free to make any necessary updates to ensure that the Dockerfile accurately reflects the needs of your application. Launch Your Application Once you have configured your FastAPI application and created a Dockerfile, now it's time for you deploy your project to Fly.io. First, make sure that you have the Fly CLI installed on your system. You can install it using the following command: curl https://getfly.fly.dev/install.sh | sh Next, navigate to your FastAPI project directory and run the following command to login to your Fly.io account: fly login You will be prompted to enter your Fly.io account email and password. Once you have logged in, you can use the following command to deploy your application: fly launch This command will automatically generate the necessary configuration files for deploying your application to Fly.io. Onces you done this you can open the public url of your application by running this command. fly open If you make any changes to your project, you can simply run the following command to deploy a new version to Fly.io: fly deploy Conclusion Deploying a FastAPI application to Fly.io is a straightforward process that allows you to host and manage your application in the cloud. By following the steps outlined in this article, you can set up a FastAPI application, configure it for deployment to Fly.io, and launch it on the platform. By using Fly.io, you can take advantage of its scalability, security, and performance features to ensure that your application is running smoothly and efficiently. Whether you are building a prototype or a full-fledged production application, Fly.io is a great choice for deploying FastAPI applications. If you encounter any issues while deploying your application to Fly.io, or if you have any questions, please do not hesitate to contact me on Twitter. Stay up-to-date on the latest web development trends Join me as we explore the exciting world of web development. In each issue, I will bring you the latest news, tutorials, and resources to help you become a top developer. Ahmad Rosid Subscribe * indicates required Email Address * your@mail.com I respect your privacy. Unsubscribe at any time. Github LinkedIn Twitter Cheatsheet Portfolio Wiki Built with Next.js, Tailwind and Vercel ©2022 All rights reserved. Ask AI to edit or generate...'''
  ## Split for every 500 characters
  input_texts = [text[i:i+500] for i in range(0, len(text), 500)]

  print('Start')
  start = time.time() 
  # Tokenize the input texts
  batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

  outputs = model(**batch_dict)

  embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
  print('pool')
  # (Optionally) normalize embeddings
  embeddings = F.normalize(embeddings, p=2, dim=1)
  print('nrom')
  scores = (embeddings[:1] @ embeddings[1:].T) * 100
  print(scores.tolist())
  end = time.time() 
  return (
      embeddings,
      end - start,
  )

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/embed")
def embed():
    (em, time) = embedding()
    return {
        "time": time,
        "size": len(em),
        "embeddings": em.tolist(),
    }


@app.get("/")
def read_root():
    return {"Hello": "World"}
