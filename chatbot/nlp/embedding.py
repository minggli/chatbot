"""
embedding

embedding module produces word embeddings

This is done using downsized (1m) GloVe vectors via NLP framework: spaCy

GloVe: Global Vectors for Word Representation
Pennington et al (2014)
"""

import spacy
import numpy as np

from collections import Counter
from itertools import chain, islice


class _BaseEmbedding(object):
    """base class for word embedding through spacy"""
    def __init__(self, raw_corpus):
        self._nlp = spacy.load('en_core_web_md')
        self.fit(raw_corpus)

    @staticmethod
    def is_sentence(iterable):
        """check if any element is not integer or string. if not then True."""
        return isinstance(iterable, list) and \
            not any(not isinstance(e, (int, str)) for e in iterable)

    @property
    def __corpus__(self):
        return self._corpus

    @__corpus__.setter
    def __corpus__(self, value):
        """index 0 reserved for out-of-vocabulary words.
        corpus consists list of documents, which consist of list of sentences,
        which consist of list of words.
        """
        try:
            assert hasattr(value, '__iter__')
            assert all(self.is_sentence(doc) for doc in value)
        except AssertionError:
            raise ValueError('corpus must be iterable containing list of'
                             'sequences')
        self._corpus = [[[word.text for word in self(sents)]
                        for sents in document] for document in value]

    def __call__(self, text):
        return self._nlp(text)

    def fit(self, raw_corpus):
        """take raw corpus and generate vocabulary with descending order by
        by most common n words"""
        self.__corpus__ = raw_corpus
        return self


class Vectorizer(_BaseEmbedding):
    """produce embedding matrix"""
    def __init__(self, raw_corpus, n=None):
        super(Vectorizer, self).__init__(raw_corpus)
        self._get_vocabulary(n)

    def _get_vocabulary(self, n):
        """vocabulary with 0 index reserved for unknown words."""
        self._ivocab = islice(zip(*Counter(chain(*chain(*self._corpus))).
                              most_common(n)), 1)
        self._word_to_ids = {word: ids for ids, word in
                             enumerate(*self._ivocab, start=1)}
        self._vocab = [word for word in self._word_to_ids]
        self._vocab.insert(0, 'UNKnown')

    def vectorize(self):
        """output embedding matrix of shape [vocab_size, n_dimensions]"""
        # !!! investigate how downsample word dimensions
        embeddings = [self(l).vector for l in self._vocab]
        return np.array(embeddings).reshape(-1, 300)


class WordEmbedding(Vectorizer):
    """encode word tokens and map with embedding matrix"""
    def __init__(self, raw_corpus, zero_pad=None, pad_length=None, n=None):
        super(WordEmbedding, self).__init__(raw_corpus, n=n)
        self.zero_pad = True if pad_length else zero_pad
        if self.zero_pad:
            self._max_length = max(len(x) for doc in self._corpus for x in doc)
            self.pad_length = pad_length or self._max_length

    def encode(self, iterable=None):
        """recursively map word with id or 0 if not in vocabulary"""
        iterable = iterable or self._corpus

        converted = list()
        for element in iterable:
            if isinstance(element, str):
                converted.append(self._word_to_ids.get(element, 0))
            elif not self.is_sentence(element):
                converted.append(self.encode(iterable=element))
            elif self.is_sentence(element):
                converted.append(self.encode(iterable=self._zero_pad(element)))
        return converted

    def _zero_pad(self, sequence):
        """pad shorter sequences with 0 (as if out of vocabulary), temporary
        solution until figure out dynamic padding (e.g. train.batch)"""
        if self.zero_pad:
            sequence.extend([0] * (self.pad_length - len(sequence)))
            return sequence[:self.pad_length]
        elif not self.zero_pad:
            return sequence
