#!/usr/bin/env bash

pip3 install --upgrade pip
pip3 install virtualenv
virtualenv venv
venv/bin/pip3 install -r requirements.txt
venv/bin/python3 -m spacy.en.download all
