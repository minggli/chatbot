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
    def __init__(self, language=None):
        self._corpus = None
        if not language:
            print('initiating NLP language pipeline...', end='', flush=True)
            self._nlp = spacy.load('en_core_web_md')
            print('done')
        elif language and isinstance(language, spacy.en.English):
            self._nlp = language

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
            assert all(self.is_sentence(sent) for sent in value)
        except AssertionError:
            raise ValueError('corpus must be iterable containing list of'
                             ' sequences.')
        self._corpus = [[[word.text for word in self(sents)]
                        for sents in document] for document in value]

        ivocab = islice(zip(*Counter(chain(*chain(*self._corpus))).
                        most_common(self._top)), 1)
        self._word2ids = {word: ids for ids, word in
                          enumerate(*ivocab, start=2)}
        self._vocab = [word for word in self._word2ids]
        self._vocab.insert(0, '|UNK|')
        self._vocab.insert(1, '|PAD|')
        self._word2ids.update({'|UNK|': 0, '|PAD|': 1})

    def __call__(self, text):
        return self._nlp(text)

    def fit(self, raw_corpus):
        """take raw corpus and generate vocabulary with descending order by
        by most common n words"""
        self.__corpus__ = raw_corpus
        return self


class Vectorizer(_BaseEmbedding):
    """produce embedding matrix"""
    def __init__(self, top=None, language=None):
        super(Vectorizer, self).__init__(language=language)
        self._top = top

    def vectorize(self):
        """output embedding matrix of shape [vocab_size, n_dimensions]"""
        # !!! investigate how downsample word dimensions
        if not self._corpus:
            raise Exception('fit corpus first.')

        zero_replace = np.full(300, 1e-8)
        embeddings = [self(l).vector if self(l).has_vector else zero_replace
                      for l in self._vocab]
        embeddings[1] = np.zeros(300)
        return np.array(embeddings, dtype=np.float32).reshape(-1, 300)


class WordEmbedding(Vectorizer):
    """encode word tokens and map with embedding matrix"""
    def __init__(self, top=None, language=None):
        super(WordEmbedding, self).__init__(top=top, language=language)

    def encode(self, zero_pad=None, pad_length=None):
        """recursively map word with id or 0 if not in vocabulary"""
        if not self._corpus:
            raise Exception('fit corpus first.')

        self._max_length = max(len(x) for doc in self._corpus for x in doc)
        self.pad_length = pad_length or self._max_length
        self.zero_pad = True if pad_length else zero_pad

        rv = [[[self._word2ids.get(word, 0) for word in self._zero_pad(sent)]
              for sent in doc] for doc in self._corpus]
        return rv

    def _zero_pad(self, sequence):
        """pad shorter sequences with 0 (as if out of vocabulary), temporary
        solution until figure out dynamic padding (e.g. train.batch)"""
        if self.zero_pad:
            copy = sequence.copy()
            copy.extend(['|PAD|'] * (self.pad_length - len(copy)))
            return copy[:self.pad_length]
        elif not self.zero_pad:
            return sequence.copy()
