"""
embedding

embedding module produces word embeddings

This is done using downsized (1m) GloVe vectors via NLP framework: spaCy

GloVe: Global Vectors for Word Representation
Pennington et al (2014)
"""

import spacy
import collections
import numpy as np


class VectorLookup:

    def __init__(self):
        self._nlp = spacy.load('en_core_web_md')

    @property
    def __corpus__(self):
        return self._corpus

    @__corpus__.setter
    def __corpus__(self, value):
        """index 0 reserved for out-of-vocabulary words."""
        if isinstance(value, collections.Iterable) and \
                not any(not isinstance(item, str) for item in value):
            tokens = [word for word in value]
            tokens.insert(0, 'UNKnown')
            self._corpus = tokens
        else:
            raise TypeError('corpus must be a iterable containing strings.')

    def __call__(self, texts):
        return self._nlp(texts)

    def fit(self, raw_corpus):
        self.__corpus__ = raw_corpus
        return self

    def transform(self):
        """output embedding of shape [vocal_size, n_dimensions]"""
        # !!! investigate how resample word dimenssions
        embeddings = [self(l).vector for l in self._corpus]
        return np.array(embeddings).reshape(-1, 300)
