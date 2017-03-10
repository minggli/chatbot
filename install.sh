sudo pip3 install --upgrade pip
sudo pip3 install virtualenv
sudo virtualenv venv
sudo venv/bin/pip3 install -r requirements.txt
sudo venv/bin/python3 -m spacy.en.download all
