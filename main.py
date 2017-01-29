import random
from settings import setting
from textmining import NHSTextMiner, NLPProcessor
import time
import pip

__author__ = 'Ming Li'


def install(package):
    """dynamically install missing package"""
    pip.main(['install', package])


try:
    from nltk.tokenize import word_tokenize
    from nltk.classify import NaiveBayesClassifier
    import nltk
except ImportError:
    install('nltk')
    from nltk.tokenize import word_tokenize
    from nltk.classify import NaiveBayesClassifier
    import nltk


nltk.download('punkt')


web_pages = {
    0: 'http://www.nhs.uk/Conditions/Heart-block/Pages/Symptoms.aspx',
    1: 'http://www.nhs.uk/conditions/frozen-shoulder/Pages/Symptoms.aspx',
    2: 'http://www.nhs.uk/conditions/coronary-heart-disease/'
    'Pages/Symptoms.aspx',
    3: 'http://www.nhs.uk/conditions/bronchitis/Pages/Symptoms-old.aspx',
    4: 'http://www.nhs.uk/conditions/warts/Pages/Introduction.aspx',
    5: 'http://www.nhs.uk/conditions/Sleep-paralysis/Pages/Introduction.aspx',
    6: 'http://www.nhs.uk/Conditions/Glue-ear/Pages/Symptoms.aspx',
    7: 'http://www.nhs.uk/Conditions/Depression/Pages/Symptoms.aspx',
    8: 'http://www.nhs.uk/Conditions/Turners-syndrome/Pages/Symptoms.aspx',
    9: 'http://www.nhs.uk/Conditions/Obsessive-compulsive-disorder/'
    'Pages/Symptoms.aspx'
}

urls = list(set(web_pages.values()))


web_scraper = NHSTextMiner(urls=urls, attrs=setting, display=True)
data = web_scraper.extract()
labels = [data[key][0] for key in data]
# cleansed_data = {key: web_scraper.cleanse(data[key]) for key in data}
nlp_processor = NLPProcessor()
processed_data = nlp_processor.process(data, {'pos': True, 'stop': True, 'lemma': True})

import spacy
nlp = spacy.load('en')
doc = nlp(processed_data['http://www.nhs.uk/Conditions/Heart-block/Pages/Symptoms.aspx'])
for token in doc:
    print(token)

# miner extracts subject, meta content (e.g. descriptwion of the page), main article


def generate_training_set(bag_of_words, label=None):
    n = 200
    sample_size = 50
    row = list()
    for i in range(n):
        row.append((web_scraper.word_feat(random.sample(bag_of_words, sample_size)), label))
    return row

feature_set = list()
mapping = dict()

for i in processed_data:
    subset = processed_data[i]
    words = word_tokenize(' '.join(subset))
    feature_set += generate_training_set(bag_of_words=words, label=subset[0])
    mapping[subset[0]] = i


clf = NaiveBayesClassifier.train(feature_set)


def decorator_converse(func):

    def t():
        time.sleep(2)
        pass

    def wrapper():

        while True:

            question = input('\nhow can I help you?')

            if len(question) == 0:
                break

            output = func(classifier=clf, question=question)

            if output and output[1] == 0:
                t()
                print('\nBased on what you told me, here is my diagnosis: {0}.'.format(output[0]))
                t()
                q = input('\nwould you like to have more information?')
                if 'yes' in q.lower():
                    print('here is the link: {0}'.format(mapping[output[0]]))

                q = input('\nwould you like to ask more questions?')
                if 'yes' in q.lower():
                    continue
                else:
                    break
            elif not output:
                t()
                print('\nSorry I am not able to help, you can improve result by asking more specific questions')
                continue
            else:
                t()
                print('\nBased on what you told me, here are several possible reasons, including: \n\n{0}'.\
                      format(output), '\n\nYou can improve result by asking more specific questions')
                t()
                q = input('\nwould you like to ask more questions?')
                if 'yes' in q.lower():
                    continue
                else:
                    break

    return wrapper


@decorator_converse
def main(classifier, question, decision_boundary=.8, limit=5):

    options = list()
    words = web_scraper.word_feat(word_tokenize(question))
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
        return ';\n'.join([i[0] + ': ({:.0%})'.format(i[1]) for i in options[:3]])
    else:
        return None

if __name__ == '__main__':
    main()

