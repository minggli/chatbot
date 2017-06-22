"""
    settings

    a repository to configure various parts of the app
"""

import os
import sys
import json
import configparser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.setrecursionlimit(30000)

CONFIGFILE = os.getenv('CONFIGFILE', default='./config.ini')
config = configparser.ConfigParser(allow_no_value=True)
config.read(CONFIGFILE)

ENGINE = os.getenv('ENGINE', default=config['GENERAL']['ENGINE'])
MAX_STEPS = int(os.getenv('STEPS',
                          default=config.getint('ENGINE', 'MAX_STEPS')))
FORCE = os.getenv('FORCE', default=config.getboolean('GENERAL', 'FORCE'))
VERBOSE = os.getenv('VERBOSE', default=config.getboolean('ENGINE', 'VERBOSE'))

WEB_BASE_URL = config['WEBDATA']['BASE_URL']
WEB_METAKEY = json.loads(config['WEBDATA']['META'])

BASE_URL = config['API']['BASE_URL']
PORT_ASK = config['API']['PORT_ASK']
PORT_SYMPTOMS = config['API']['PORT_SYMPTOMS']

MAX_WORDS = int(config['ENGINE']['MAX_WORDS']) \
            if config['ENGINE']['MAX_WORDS'] else None
BATCH_SIZE = config.getint('ENGINE', 'BATCH_SIZE')
STATE_SIZE = config.getint('ENGINE', 'STATE_SIZE')
STEP_SIZE = config.getint('ENGINE', 'STEP_SIZE')

NLP_ATTRS = json.loads(config['NLP']['PROCESS'])
NLP_CONTRACTIONS = json.loads(config['NLP']['CONTRACTIONS'])

APP_CONFIG = {
    'SECRET_KEY': '\x9c\xedh\xdf\x8dS\r\xe3]\xc3\xd3\xbd\x0488\xfc\xa6<\xfe'
                  '\x94\xc8\xe0\xc7\xdb',
    'SESSION_COOKIE_NAME': 'chatbot_session',
    'DEBUG': False
}


class CacheSettings(object):

    path = config['GENERAL']['DATA_LOCATION']
    index = path + 'index_pages.pkl'
    symptoms = path + 'symptoms.pkl'
    processed_data = path + 'nlp_data.pkl'

    @classmethod
    def check(CacheSettings, filename):
        return True if os.path.exists(filename) else False


def build_config(filename):
    """create a configparser object and write a ini file."""
    config = configparser.ConfigParser(allow_no_value=True)

    config['GENERAL'] = dict()
    config['GENERAL']['ENGINE'] = 'TENSORFLOW'
    config['GENERAL']['FORCE'] = 'false'
    config['GENERAL']['DATA_LOCATION'] = 'cache/'
    config['API'] = dict()
    config['API']['BASE_URL'] = '/chatbot/api/v1'
    config['API']['PORT_ASK'] = '5000'
    config['API']['PORT_SYMPTOMS'] = '5001'
    config['ENGINE'] = dict()
    config['ENGINE']['MAX_WORDS'] = ''
    config['ENGINE']['BATCH_SIZE'] = '500'
    config['ENGINE']['STATE_SIZE'] = '128'
    config['ENGINE']['STEP_SIZE'] = '40'
    config['ENGINE']['MAX_STEPS'] = '5000'
    config['ENGINE']['VERBOSE'] = 'false'
    config['WEBDATA'] = dict()
    config['WEBDATA']['BASE_URL'] = 'http://www.nhs.uk'
    config['WEBDATA']['META'] = """
                                {"desc_attributes": {"name": "description"},
                                "subj_attributes": {"name": "DC.title"},
                                "article_attributes": {
                                                    "start_t_0": "Overview",
                                                    "start_t_1": "Print:",
                                                    "start_t_2": "",
                                                    "end_t_0": "Share:",
                                                    "end_t_1": "",
                                                    "end_t_2": ""}
                                }"""

    config['NLP'] = dict()
    config['NLP']['PROCESS'] = """
        {"pipeline":
                    {"pos": true, "stop": true, "lemma": true},
         "part_of_speech_exclude":
                    ["ADP", "PUNCT", "DET", "CONJ", "PART", "PRON", "SPACE"]
        }"""

    config['NLP']['CONTRACTIONS'] = """
                                    {"ain't": "am not",
                                    "aren't": "are not",
                                    "can't": "cannot",
                                    "'cause": "because",
                                    "could've": "could have",
                                    "couldn't": "could not",
                                    "couldn't've": "could not have",
                                    "didn't": "did not",
                                    "doesn't": "does not",
                                    "don't": "do not",
                                    "hadn't": "had not",
                                    "hadn't've": "had not have",
                                    "hasn't": "has not",
                                    "haven't": "have not",
                                    "he'd": "he would",
                                    "he'd've": "he would have",
                                    "he'll": "he will",
                                    "he'll've": "he will have",
                                    "he's": "he has",
                                    "how'd": "how did",
                                    "how'd'y": "how do you",
                                    "how'll": "how will",
                                    "how's": "how is",
                                    "i'd": "I would",
                                    "i'd've": "I would have",
                                    "i'll": "I will",
                                    "i'll've": "I will have",
                                    "i'm": "I am",
                                    "i've": "I have",
                                    "isn't": "is not",
                                    "it'd": "it would",
                                    "it'd've": "it would have",
                                    "it'll": "it will",
                                    "it'll've": "it will have",
                                    "it's": "it is",
                                    "let's": "let us",
                                    "ma'am": "madam",
                                    "mayn't": "may not",
                                    "might've": "might have",
                                    "mightn't": "might not",
                                    "mightn't've": "might not have",
                                    "must've": "must have",
                                    "mustn't": "must not",
                                    "mustn't've": "must not have",
                                    "needn't": "need not",
                                    "needn't've": "need not have",
                                    "o'clock": "of the clock",
                                    "oughtn't": "ought not",
                                    "oughtn't've": "ought not have",
                                    "shan't": "shall not",
                                    "sha'n't": "shall not",
                                    "shan't've": "shall not have",
                                    "she'd": "she would",
                                    "she'd've": "she would have",
                                    "she'll": "she will",
                                    "she'll've": "she will have",
                                    "she's": "she has",
                                    "should've": "should have",
                                    "shouldn't": "should not",
                                    "shouldn't've": "should not have",
                                    "so've": "so have",
                                    "so's": "so is",
                                    "that'd": "that would",
                                    "that'd've": "that would have",
                                    "that's": "that is",
                                    "there'd": "there had",
                                    "there'd've": "there would have",
                                    "there's": "there is",
                                    "they'd": "they would",
                                    "they'd've": "they would have",
                                    "they'll": "they will",
                                    "they'll've": "they will have",
                                    "they're": "they are",
                                    "they've": "they have",
                                    "to've": "to have",
                                    "wasn't": "was not",
                                    "we'd": "we would",
                                    "we'd've": "we would have",
                                    "we'll": "we will",
                                    "we'll've": "we will have",
                                    "we're": "we are",
                                    "we've": "we have",
                                    "weren't": "were not",
                                    "what'll": "what will",
                                    "what'll've": "what will have",
                                    "what're": "what are",
                                    "what's": "what is",
                                    "what've": "what have",
                                    "when's": "when is",
                                    "when've": "when have",
                                    "where'd": "where did",
                                    "where's": "where is",
                                    "where've": "where have",
                                    "who'll": "who will",
                                    "who'll've": "who will have",
                                    "who's": "who is",
                                    "who've": "who have",
                                    "why's": "why is",
                                    "why've": "why have",
                                    "will've": "will have",
                                    "won't": "will not",
                                    "won't've": "will not have",
                                    "would've": "would have",
                                    "wouldn't": "would not",
                                    "wouldn't've": "would not have",
                                    "y'all": "you all",
                                    "y'all'd": "you all would",
                                    "y'all'd've": "you all would have",
                                    "y'all're": "you all are",
                                    "y'all've": "you all have",
                                    "you'd": "you would",
                                    "you'd've": "you would have",
                                    "you'll": "you will",
                                    "you'll've": "you will have",
                                    "you're": "you are",
                                    "you've": "you have"
                                    }"""

    with open(CONFIGFILE, 'w') as f:
        config.write(f)
