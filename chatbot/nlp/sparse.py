"""
    sparse

    NLP processor
"""

import spacy
import pickle

from chatbot.settings import CacheSettings


class NLPProcessor:
    """using SpaCy's features to extract relevance out of raw texts."""

    def __init__(self, attrs):
        print('initiating NLP language pipeline...', end='', flush=True)
        self._nlp = spacy.load('en_core_web_md')
        print('done')

        self._is_string = None
        self._is_list = None
        self._output = None
        self._doc_object = None
        self._content = None
        self._attrs = attrs

    def process(self, content):

        if isinstance(content, str):
            self._is_string = True
            self._doc_object = self._nlp(content)
        elif isinstance(content, list):
            self._is_list = True
            self._content = [self._nlp(' '.join(doc)) for doc in content]
        else:
            raise TypeError('require string or dictionary.')

        if self._is_string:
            processed = self._pipeline(doc_object=self._doc_object)
            self._output = ' '.join(processed.text.split())
            return self._output

        elif self._is_list:
            if not CacheSettings.check(CacheSettings.processed_data):
                print('Using NLP language pipeline to process...', end='',
                      flush=True)
                self._output = [' '.join(
                                self._pipeline(doc_object=doc).text.split())
                                for doc in self._content]
                print('done')
                with open(CacheSettings.processed_data, 'wb') as f:
                    pickle.dump(self._output, f)
                return self._output
            else:
                with open(CacheSettings.processed_data, 'rb') as f:
                    self._output = pickle.load(f)
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
        assert isinstance(
            doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
        return self._nlp(
            ' '.join([token.text for token in doc_object
                      if token.pos_ not in parts])) if switch else doc_object

    def __stop_word__(self, doc_object, switch=True):
        """only remove stops when not part of phrase e.g. back pain.
        """
        assert isinstance(
            doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
        noun_chunks = ' '.join(
            set([phrase.text for phrase in doc_object.noun_chunks]))
        return self._nlp(
            ' '.join([token.text for token in doc_object
                      if token.is_stop is False or
                      token.text in noun_chunks])) if switch else doc_object

    def __lemmatize__(self, doc_object, switch=True):
        assert isinstance(
            doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
        return self._nlp(
            ' '.join([token.lemma_
                     for token in doc_object])) if switch else doc_object
