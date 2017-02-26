import random

from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier

from ..helpers import NHSTextMiner


def wrapper_classifier(func):
    def wrapper(*args, **kwargs):
        print('training classifier...', end='\n', flush=True)
        trained_clf = NaiveBayesClassifier.train(func(*args, **kwargs))
        print('done', flush=True)
        return trained_clf
    return wrapper


@wrapper_classifier
def train_model(input_data, label, n=100):
    # TODO investigate better algorithm e.g. LDA and Reinforcement NN
    print('starting to generate training data...', end='', flush=True)
    shuffled_feature_set = list()
    for key in input_data:
        words = word_tokenize(input_data[key])
        row = [tuple((NHSTextMiner.word_feat(random.sample(
            words, 100)), label[key])) for r in range(n)]
        shuffled_feature_set += row
    return shuffled_feature_set


def nb_classifier(query, engine, nlp, decision_boundary=.8, limit=5):
    """spell out most probable diseases and respective percentages."""
    options = list()
    words = NHSTextMiner.word_feat(word_tokenize(nlp.process(query)))
    print('understanding {}...'.format(words))
    objects = engine.prob_classify(words)
    keys = list(objects.samples())

    for key in keys:
        prob = objects.prob(key)
        options.append((key, prob))

    options.sort(key=lambda x: x[1], reverse=True)
    options = options[:limit]

    if options[0][1] > decision_boundary:
        return '{0} ({1:.0%})'.format(options[0][0], options[0][1]), 0
    elif options[0][1] > decision_boundary / 3:
        return ';\n'.join([pair[0] + ' ({:.0%})'.format(pair[1])
                           for pair in options]), 1
    else:
        return None
