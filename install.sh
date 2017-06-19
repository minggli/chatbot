#!/usr/bin/env bash

pip3 install --upgrade pip
sudo pip3 install virtualenv
virtualenv venv
venv/bin/pip3 install -r requirements.txt
venv/bin/python3 -m spacy download en_core_web_md
