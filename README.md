# chatbot
a naive chatbot that sometimes misdiagnoses.

This retrieval-based prototype bot uses publicly available health information to generate indicative diagnosis and link to NHS leaflet. The result is not diagnosis and non-clinical use only.

![alt text](screenshots/example_cold.png "example common cold")

## Requirement
Python >= 3.4  
Virtualenv >= 15.1 or Docker

## Installation
### Virtualenv
Run script `./install.sh`. This will first make a virtual environment `venv` and install components within it.
### Docker
Run `docker make -t chatbot .` to make a Docker image using Debian base image.

## Running chatbot

`export ENGINE=NLTK` (default) to use NLTK backend for traditional Bag of Words model with Naive Bayes  

`export ENGINE=TENSORFLOW` to use Tensorflow backend for representational sequence classification with Long-short Term Memroy (LSTM)

### Virtualenv
Within virtual environment `venv` (`source venv/bin/activate`), you can launch separate service:  
`python3 -m chatbot.services.ask`  

`python3 -m chatbot.services.symptoms`
### Docker
Run docker image and instantiate docker container for each service:  
`docker run -p 5000:5000 -e ENGINE=$ENGINE $(docker images "chatbot" -q) python -m chatbot.services.ask`  

`docker run -p 5001:5001 $(docker images "chatbot" -q) python -m chatbot.services.symptoms`  

For the first time running either service, it will take longer than usual as it needs to download, process and cache web data.

## API endpoints
It accepts payload as simple as `{"questions": "your questions or description of symptoms"}` to below to query for indicative diagnosis:  
`POST /chatbot/api/v1/ask`  

To list all leaflets or leaflet for the chosen symptom:  

`GET /chatbot/api/v1/symptoms` or
`GET /chatbot/api/v1/symptoms/<string:symptom_name>`


## Further development ideas
~~Using word vector rather than sparse matrix to extract semantic proximity in embedded space;~~

Explore Recurrent Neural Network (e.g. LSTM) to move from retrieval-based model to generative;

~~Explore supervised Latent Dirichlet Allocation, RNN as text classification alternative to Naive Bayes.~~
