"""
    naivebayes

    using naive bayes as probablistic classifer with inherent assumption of
    indepedence of features.
"""

import nltk
import random

from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier

from chatbot.engine import TextMiner, corpus, labels
from chatbot.nlp.sparse import NLPPipeline
from chatbot.serializers import feed_conversation
from chatbot.settings import NLP_ATTRS

nltk.download('punkt')


def word_feat(words):
    """traditional atomic tokenization"""
    return dict([(word, True) for word in words])


def resample(feat, label, sample_size, n=100):
    """bootstrap resampling of original sample."""
    k = int(sample_size * len(feat))
    return ((word_feat(random.sample(feat, k)), label) for _ in range(n))


def train_model(documents, labels, sample_size=.3, verbose=True):

    if verbose:
        print('starting to generate training data...', end='', flush=True)

    labeled_feature_set = list()
    for n, doc in enumerate(documents):
        feature = word_tokenize(' '.join(doc))
        label = labels[n]
        labeled_feature_set.extend(resample(feature, label, sample_size, n=50))

    if verbose:
        print('done', flush=True)
        print('training model...this may take a few minutes.',
              flush=True, end='')

    trained_model = NaiveBayesClassifier.train(iter(labeled_feature_set))

    if verbose:
        print('done', flush=True)
    return trained_model


def preprocess(q):
    return word_feat(
           word_tokenize(nlp.process(TextMiner.split_contraction(q))))


nlp = NLPPipeline(attrs=NLP_ATTRS)
processed_data = nlp.process(corpus)
engine = train_model(processed_data, labels, sample_size=0.3)


def classify(query,
             engine=engine,
             threshold=.85,
             limit=5):
    """spell out most probable diseases and respective percentages."""
    words = preprocess(' '.join(query))
    print('understanding {}...'.format(words))
    objects = engine.prob_classify(words)
    keys = list(objects.samples())

    samples = [tuple((key, objects.prob(key))) for key in keys]

    return feed_conversation(samples, limit, threshold)
