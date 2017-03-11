"""
    settings

    a repository to configure various parts of the app
"""

NHS_BASE_URL = 'http://www.nhs.uk'
API_BASE_URL = '/chatbot/api/v1'
DATA_LOC = 'app/cache/'

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

NLP = {
    'pipeline': {
        'pos': True,
        'stop': True,
        'lemma': True
    },
    'part_of_speech_exclude': ['ADP', 'PUNCT', 'DET', 'CONJ', 'PART', 'PRON', 'SPACE']
}
