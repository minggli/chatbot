"""
embedding

embedding module produces word embeddings

This is done using downsized (1m) GloVe vectors via NLP framework: spaCy

GloVe: Global Vectors for Word Representation
Pennington et al (2014)
"""

import spacy
import numpy as np

from itertools import chain, islice
from collections import Counter, Sequence


class _BaseEmbedding(object):
    """base class for word embedding through spacy"""
    def __init__(self, raw_corpus):
        self._nlp = spacy.load('en_core_web_md')
        self.fit(raw_corpus)

    @staticmethod
    def is_sentence(obj):
        """check if any element is not integer or string. if not then True."""
        return isinstance(obj, Sequence) and not isinstance(obj, str) and \
            not any(not isinstance(e, (int, str)) for e in obj)

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
        self._word2ids = {word: ids for ids, word in
                          enumerate(*self._ivocab, start=1)}
        self._vocab = [word for word in self._word2ids]
        self._vocab.insert(0, 'UNKnown')
        self._word2ids.update({'UNKnown': 0})

    def vectorize(self):
        """output embedding matrix of shape [vocab_size, n_dimensions]"""
        # !!! investigate how downsample word dimensions
        embeddings = [self(l).vector for l in self._vocab]
        return np.array(embeddings).reshape(-1, 300)


class WordEmbedding(Vectorizer):
    """encode word tokens and map with embedding matrix"""
    def __init__(self, raw_corpus, zero_pad=None, pad_length=None, n=None):
        super(WordEmbedding, self).__init__(raw_corpus, n=n)

        self._max_length = max(len(x) for doc in self._corpus for x in doc)
        self.pad_length = pad_length or self._max_length
        self.zero_pad = True if pad_length else zero_pad

    def encode(self):
        """recursively map word with id or 0 if not in vocabulary"""
        rv = [[[self._word2ids.get(word, 0) for word in self._zero_pad(sent)]
              for sent in doc] for doc in self._corpus]
        return rv

    def _zero_pad(self, sequence):
        """pad shorter sequences with 0 (as if out of vocabulary), temporary
        solution until figure out dynamic padding (e.g. train.batch)"""
        if self.zero_pad:
            sequence.extend(['UNKnown'] * (self.pad_length - len(sequence)))
            return sequence[:self.pad_length]
        elif not self.zero_pad:
            return sequence
