import random
from settings import setting
from textmining import NHSTextMining
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

urls = list(web_pages.values())

web_scraper = NHSTextMining(urls=urls, attrs=setting, n=None, display=True)
data = web_scraper.extract()

# miner extracts subject, meta content (e.g. description of the page), main article


def preview_data():
    for i in range(1, 10):
        print(data[web_pages[i]][:1])
        input('press enter to continue...')


def word_feat(words):
    return dict([(word, True) for word in words])


def generate_training_set(bag, target=None):
    n = 200
    sample_size = 50
    row = list()
    for i in range(n):
        row.append((word_feat(random.sample(bag, sample_size)), target))
    return row

feature_set = list()
mapping = dict()

for i in web_pages.values():
    try:
        subset = data[i]
    except KeyError:
        pass

    words = word_tokenize(' '.join(NHSTextMining.cleanse(subset)))
    feature_set = feature_set + generate_training_set(bag=words, target=subset[0])
    mapping[subset[0]] = i

print(feature_set[0][0])

classifier = NaiveBayesClassifier.train(feature_set)


def classify(question, decision_boundary=.8):

    options = list()
    words = word_feat(word_tokenize(question))
    obj = classifier.prob_classify(words)
    keys = list(obj.samples())

    for i in keys:

        prob = obj.prob(i)
        options.append((i, prob))

    options.sort(key=lambda x: x[1], reverse=True)

    if options[0][1] > decision_boundary:
        return obj.max(), 0
    elif options[0][1] > decision_boundary / 3:
        return ';\n'.join([i[0] + ': ({:.0%})'.format(i[1]) for i in options[:3]])
    else:
        return None


def converse(s=2):

    t = time.sleep(s)

    while True:

        t
        question = input('\nhow can I help you?')

        if len(question) == 0:
            break

        output = classify(question)

        if output and output[1] == 0:
            t
            print('\nBased on what you told me, here is my diagnosis: {0}.'.format(output[0]))
            t
            q = input('\nwould you like to have more information?')
            if 'yes' in q.lower():
                print('here is the link: {0}'.format(mapping[output[0]]))

            q = input('\nwould you like to ask more questions?')
            if 'yes' in q.lower():
                continue
            else:
                break
        elif not output:
            print('\nSorry I am not able to help, you can improve result by asking more specific questions')
            t
            continue
        else:
            print('\nBased on what you told me, here are several possible reasons, including: \n{0}'.\
                  format(output), '\nYou can improve result by asking more specific questions')
            t
            q = input('\nwould you like to ask more questions?')
            if 'yes' in q.lower():
                continue
            else:
                break

if __name__ == '__main__':
    converse()

