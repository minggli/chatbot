# setting file

"""keys to detect useful information and main article of web pages."""

NHS_BASE_URL = 'http://www.nhs.uk'
API_BASE_URL = '/chatbot/api/v1'
DATA_LOC = 'cache/'

TEXTMINER = {
    'desc_attributes': {
        'name': 'description'
    },
    'subj_attributes': {
        'name': 'DC.title'
    },
    'article_attributes': {
        'start_t_0': 'Overview',
        'start_t_1': 'Print this page',
        'start_t_2': 'Print this page',
        'end_t_0': 'Share:',
        'end_t_1': '',
        'end_t_2': ''
    }
}

NLP_PROCESSOR = {
        'pipeline': {'pos': True, 'stop': True, 'lemma': True},
        'part_of_speech_include': {'ADJ', 'DET', 'ADV', 'ADP', 'VERB', 'NOUN', 'PART'}
        # 'part_of_speech_include': {'ADJ', 'DET', 'ADV', 'SPACE', 'CONJ', 'PRON', 'ADP', 'VERB', 'NOUN', 'PART'}
}
