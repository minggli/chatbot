pip3 install virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
source venv/bin/activate
python3 -m spacy.en.download all
source venv/bin/activate
