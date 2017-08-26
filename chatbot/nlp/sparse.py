"""
    sparse

    NLP pipeline
"""

import pickle

from chatbot.nlp import ifninstall, _BaseNLP
from chatbot.settings import CacheSettings


class NLPPipeline(_BaseNLP):
    """using SpaCy's features to extract relevance out of raw texts."""

    def __init__(self, attrs):
        ifninstall(self.__class__.sm_pkg)
        print('initiating NLP language pipeline...', end='', flush=True)
        import en_core_web_sm
        self._nlp = en_core_web_sm.load()
        print('done')

        self._is_string = None
        self._is_list = None
        self._output = None
        self._doc_object = None
        self._content = None
        self._attrs = attrs

    def process(self, content, prod=False):

        if isinstance(content, str):
            self._is_string = True
            self._doc_object = self._nlp(content)
        elif isinstance(content, list):
            self._is_list = True
            if not CacheSettings.check(CacheSettings.processed_data) or prod:
                print('using NLP language pipeline to process...', end='',
                      flush=True)
                self._content = [[s for chunk in doc for s in
                                  self._nlp(' '.join(chunk.split())).sents]
                                 for doc in content]
        else:
            raise TypeError('require string or dictionary.')

        if self._is_string:
            processed = self._pipeline(doc_object=self._doc_object)
            self._output = ' '.join(processed.text.split())
            return self._output

        elif self._is_list:
            if not CacheSettings.check(CacheSettings.processed_data) or prod:

                self._output = [[self._pipeline(sent).text for sent in doc]
                                for doc in self._content]
                self._output = self._limitlengh(self._output, limit=1)

                print('done')
                if not prod:
                    with open(CacheSettings.processed_data, 'wb') as f:
                        pickle.dump(self._output, f)
                return self._output
            else:
                print('fetching cached NLP processed data...', end='',
                      flush=True)
                with open(CacheSettings.processed_data, 'rb') as f:
                    self._output = pickle.load(f)
                print('done')
                return self._output

    def _pipeline(self, doc_object):
        return self.__lemmatize__(self.__stop_word__(self.__part_of_speech__(
            doc_object, parts=self._attrs['part_of_speech_exclude'],
            switch=self._attrs['pipeline']['pos']),
            switch=self._attrs['pipeline']['stop']),
            switch=self._attrs['pipeline']['lemma'])

    def __part_of_speech__(self, doc_object, parts, switch=True):
        """filter unrelated parts of speech (POS) and return required parts
        """
        return self._nlp(
            ' '.join([token.text for token in doc_object
                      if token.pos_ not in parts])) if switch else doc_object

    def __stop_word__(self, doc_object, switch=True):
        """only remove stops when not part of phrase e.g. back pain.
        """
        noun_chunks = ' '.join(
            set([np.text for np in self._nlp(doc_object.text).noun_chunks]))
        return self._nlp(' '.join(
                    [token.text for token in doc_object if token.is_stop
                     is False or token.text in noun_chunks])
                     ) if switch else doc_object

    def __lemmatize__(self, doc_object, switch=True):
        return self._nlp(
            ' '.join([token.lemma_
                     for token in doc_object])) if switch else doc_object

    def _limitlengh(self, it, limit=1):
        return [[sent for sent in d if len(sent.split()) > limit] for d in it]
