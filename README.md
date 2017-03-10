# chatbot
a naive chatbot that sometimes misdiagnoses. 

This retrieval-based bot uses publicly available health information and a set of NLP techniques to generate indicative diagnosis and NHS leaflet for that diagnosis. The result is not to be treated as medical advice.

## Installation
Run `./install.sh` at chatbot/. This will create a virtualenv instance and install all components required.

## Running chatbot
Within virtualenv venv (`source venv/bin/activate`), you can launch separate service: 

`python3 -m app.services.ask`

`python3 -m app.services.symptoms`

It takes approx. 10 minutes to launch `ask` but for the first time, it will take longer as it needs to download, process and cache web data.

## API Endpoints
Following endpoints are available to consume:

`POST /chatbot/api/v1/ask`

It accepts payload as simple as `{"questions": "your questions or description of symptoms"}`


`GET /chatbot/api/v1/symptoms`
`GET /chatbot/api/v1/symptoms/<string:symptom_name>`

To list all leaflets or leaflet for the chosen symptom.
