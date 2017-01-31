from settings import setting
from get_urls import web_pages
from textmining import NHSTextMiner, NLPProcessor
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
import nltk
import random
import time
import os
import sys
import pickle
sys.setrecursionlimit(30000)

__author__ = 'Ming Li'

web_scraper = NHSTextMiner(urls=sorted(list(web_pages.values())), attrs=setting, display=True)
data = web_scraper.extract()
labels = {key: data[key][0] for key in data}
mapping = {v: k for k, v in labels.items()}
nlp_processor = NLPProcessor(attrs=setting)

if not os.path.exists('data/processed_data.pkl'):
    processed_data = nlp_processor.process(data)
    with open('data/processed_data.pkl', 'wb') as filename:
        pickle.dump(processed_data, filename)
else:
    with open('data/processed_data.pkl', 'rb') as filename:
        processed_data = pickle.load(filename)


def generate_training_set(data, n=100):

    print('starting to generate training data...', end='', flush=True)
    shuffled_feature_set = list()
    for key in data:
        words = word_tokenize(data[key])
        row = [tuple((web_scraper.word_feat(random.sample(words, 100)), labels[key])) for r in range(n)]
        shuffled_feature_set += row
    print('done', flush=True)
    return shuffled_feature_set


def train_classifier(feature_set):
    print('training classifier...', end='', flush=True)
    trained_clf = NaiveBayesClassifier.train(feature_set)
    print('done', flush=True)
    return trained_clf

clf = train_classifier(feature_set=generate_training_set(processed_data))


def decorator_converse(func):

    def t(s=2):
        time.sleep(s)
        pass

    def wrapper(ambiguity_trials=3):

        os.system('clear')
        aggregate_text = list()
        count = 0

        while True:

            if count == 0:
                q_string = '\n\nHow can I help you?'
                aggregate_text = list()

            elif count == ambiguity_trials:
                t(s=1)
                print('\nOk, we don\'t seem to get anywhere. Let\'s start again...')
                count = 0
                t()
                continue

            else:
                q_string = '\n\nCan you tell me more about the symptoms?'

            t(s=1)
            question = input(q_string)

            if 'exit' in question.lower():
                sys.exit()

            aggregate_text.append(question)

            output = func(classifier=clf, question=' '.join(aggregate_text))

            if output and output[1] == 0:
                t()
                print('\nBased on what you told me, here is my diagnosis: {0}.'.format(output[0]))
                t()
                q = input('\nwould you like to have NHS leaflet?')
                if 'yes' in q.lower():
                    print('here is the link: {0}'.format(mapping[output[0]]))
                count = 0

            elif not output:
                t()
                print('\nSorry I don\'t have enough knowledge to help you, you can improve result by describing symptoms further.')
                count += 1

            else:
                t()
                print('\nBased on what you told me, here are several possible reasons, including: \n\n{0}'.\
                      format(output), '\n\nYou can improve result by describing symptoms further.')
                count += 1

    return wrapper


@decorator_converse
def main(classifier, question, decision_boundary=.8, limit=5):

    options = list()
    words = web_scraper.word_feat(word_tokenize(nlp_processor.process(question)))
    print('understanding {}...'.format(words))
    obj = classifier.prob_classify(words)
    keys = list(obj.samples())

    for key in keys:

        prob = obj.prob(key)
        options.append((key, prob))

    options.sort(key=lambda x: x[1], reverse=True)
    options = options[:limit]

    if options[0][1] > decision_boundary:
        return obj.max(), 0
    elif options[0][1] > decision_boundary / 3:
        return ';\n'.join([pair[0] + ': ({:.0%})'.format(pair[1]) for pair in options])
    else:
        return None

if __name__ == '__main__':
    main()
