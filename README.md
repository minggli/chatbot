# chatbot
a naive chatbot that sometimes misdiagnoses. 

This retrival-based bot uses publicly available health information and a set of NLP techniques to generate indicative diagnosis and NHS leaflet for that diagnosis. The result is not to be treated as medical advice.

## Virtualenv Installation
You will need [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) to create the environment underpinning this chatbot.

First, run `virtualenv venv` within the directory containing chatbot.

Second, activate virtualenv `source venv/bin/activate`

Finally, install required environments `pip install -r requirements.txt`

## API Endpoints
Following endpoints are available to consume:

`GET /chatbot/api/v1/symptoms`


`GET /chatbot/api/v1/symptoms/<string:symptom_name>`


`POST /chatbot/api/v1/ask`, it accepts a simple payload `{"questions": "your questions or description of payload"}`
