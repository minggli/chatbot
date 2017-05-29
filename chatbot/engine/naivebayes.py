"""
    naivebayes

    using naive bayes as probablistic classifer with inherent assumption of
    indepedence of features.
"""

import nltk
import random

from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier

from . import NLPProcessor, NLP, corpus, labels

nltk.download('punkt')


def word_feat(words):
    """traditional atomic tokenization"""
    return dict([(word, True) for word in words])


def resample(feat, label, sample_size, n=100):
    k = int(sample_size * len(feat))
    return [(word_feat(random.sample(feat, k)), label) for _ in range(n)]


def train_model(documents, labels, sample_size=.3, verbose=True):

    if verbose:
        print('starting to generate training data...', end='', flush=True)

    labeled_feature_set = list()
    for n, doc in enumerate(documents):
        feature = word_tokenize(doc)
        label = labels[n]
        resampled = resample(feature, label, sample_size)
        labeled_feature_set += resampled

    if verbose:
        print('done', flush=True)
        print('training model...this may take a few minutes.',
              flush=True, end='')

    trained_model = NaiveBayesClassifier.train(labeled_feature_set)

    if verbose:
        print('done', flush=True)
    return trained_model


def naive_bayes_classifier(query, engine, decision_boundary=.85, limit=5):
    """spell out most probable diseases and respective percentages."""
    words = word_feat(word_tokenize(nlp.process(query)))
    print('understanding {}...'.format(words))
    objects = engine.prob_classify(words)
    keys = list(objects.samples())

    options = [tuple((key, objects.prob(key))) for key in keys]
    options.sort(key=lambda x: x[1], reverse=True)
    options = options[:limit]

    if options[0][1] > decision_boundary:
        return options[0]
    elif options[0][1] > decision_boundary / 3:
        return options
    else:
        return None


nlp = NLPProcessor(attrs=NLP)
processed_data = nlp.process(corpus)
engine = train_model(processed_data, labels, sample_size=0.3)
