import pip
import spacy
import pickle
import os
import sys
import requests
from multiprocessing import Pool
from bs4 import BeautifulSoup

sys.setrecursionlimit(30000)

__author__ = 'Ming Li'

# web scrapping module for NHS symptoms

class NHSTextMiner(object):

    """web scrapping module using BeautifulSoup4 and Requests"""

    def __init__(self, urls, attrs, display=False):

        """urls and attrs to be supplied by main and setting."""

        assert isinstance(urls, list), 'require a list of urls'
        assert isinstance(attrs, dict), 'attributes must be a dictionary'

        urls.append('http://www.nhs.uk/conditions/frozen-soulder')
        urls.append('http://www.nhs.uk/conditions/whipla')
        

        self._urls = urls
        self._failed_urls = list()
        self._attrs = attrs
        self._count = len(urls)
        self._soups = list()
        self._display = display
        self._output = dict()

    def _get(self, url):

        """get all web pages and create soup objects ready for information extraction"""

        r = requests.get(url=url)

        if self._display:
            print(r.status_code, r.url)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html5lib')
            return soup
        else:
            self._failed_urls.append(url)
            print(len(self._failed_urls))

         
    def _cache_get(self):

        if not os.path.exists('data/symptom_pages.pkl'):

            if self._display:
                print('{0} pages are being downloaded...'.format(len(self._urls)), flush=True, end='\n')

            with Pool(10) as p:
                self._soups = p.map(self._get, self._urls)

            print(self._failed_urls)
            for f_url in self._failed_urls:
                self._urls.remove(f_url)
                self._count -= 1

            print(len(self._soups), len(self._failed_urls))
            sys.exit()

            with open('data/symptom_pages.pkl', 'wb') as filename:
                pickle.dump(self._soups, filename)
            with open('data/symptom_urls.pkl', 'wb') as filename:
                pickle.dump(self._urls, filename)
        else:
            with open('data/symptom_pages.pkl', 'rb') as filename:
                self._soups = pickle.load(filename)
            with open('data/symptom_urls.pkl', 'rb') as filename:
                self._urls = pickle.load(filename)


    def extract(self):

        """get all web pages and create soup objects ready for information extraction"""

        self._cache_get()

        print('starting to extract information from websites...', flush=True, end='')
        
        self._failed_urls.clear()

        for i, page_url in enumerate(self._urls):

            page = self._soups[i]

            try:

                subj = page.find('meta', attrs=self._attrs['subj_attributes']).get('content')
                meta = page.find('meta', attrs=self._attrs['desc_attributes']).get('content')
                article = [element.get_text(strip=True) for element in page.find_all(['p', 'li', 'meta'])]
            
            except AttributeError:
                self._failed_urls.append(page_url)
                continue

            subj = subj.replace(' - NHS Choices', '')

            start_idx = int()
            end_idx = int()

            for j, value in enumerate(article):

                # using 3 keys each end to identify range of main article
                try:

                    s1 = article[j] == self._attrs['article_attributes']['start_t_2']
                    s2 = article[j + 1] == self._attrs['article_attributes']['start_t_1']
                    s3 = article[j + 2] == self._attrs['article_attributes']['start_t_0']
                    e1 = article[j] == self._attrs['article_attributes']['end_t_0']
                    e2 = article[j + 1] == self._attrs['article_attributes']['end_t_1']
                    e3 = article[j + 2] == self._attrs['article_attributes']['end_t_2']
                
                except IndexError:
                    self._failed_urls.append(page_url)
                    break

                if s1 and s2 and s3:
                    start_idx = j + 2

                if start_idx and e1 and e2 and e3:
                    end_idx = j
                    break

            content = article[start_idx: end_idx]

            if len(content) < 5:
                self._failed_urls.append(page_url)
                continue

            content.insert(0, subj)
            content.insert(1, meta)

            self._output[page_url] = content

        for f_url in list(set(self._failed_urls)):
            self._urls.remove(f_url)
            self._count -= 1

        print('done. {} of {} failed to be extracted.'.format(len(set(self._failed_urls)), len(self._soups)), flush=True)

        return self._output

    @staticmethod
    def cleanse(words, removals='''!"#$%&()*+/;<=>?@[\]^_`{|}~.,:'''):
        return [word.encode('utf-8').decode('ascii', 'ignore').translate(str.maketrans(removals, ' '*len(removals))).replace('\xa0', ' ') for word in words]

    @staticmethod
    def word_feat(words):
        return dict([(word, True) for word in words])


class AdditiveDict(dict):

    def __init__(self, iterable=None):
        if not iterable:
            pass
        else:
            assert hasattr(iterable, '__iter__')
            for i in iterable:
                self.__setitem__(i[0], 0)

    def __missing__(self, key):
        return 0

    def __setitem__(self, key, value):
        super(AdditiveDict, self).__setitem__(key, self.__getitem__(key) + 1)


class NLPProcessor(object):
    """using SpaCy's features to extract relevance out of raw texts."""

    def __init__(self, attrs):
        """takes in raw_string or dictionary resulted from NHSTextMiner"""
        print('initiating SpaCy\'s NLP language pipeline...', end='', flush=True)
        self._nlp = spacy.load('en')
        print('done')

        self._is_string = None
        self._is_dict = None
        self._output = None
        self._doc_object = None
        self._content = None
        self._attrs = attrs

    def process(self, content):

        if isinstance(content, str):
            self._is_string = True
            self._doc_object = self._nlp(content)
        elif isinstance(content, dict):
            self._is_dict = True
            self._content = {key: self._nlp(' '.join(content[key])) for key in content}
        else:
            raise TypeError('require string or dictionary.')

        if self._is_string:
            processed = self._pipeline(doc_object=self._doc_object)
            self._output = ' '.join(processed.text.split())
            return self._output

        elif self._is_dict:
            print('Using SpaCy\'s NLP language pipeline to process...', end='', flush=True)
            for document in self._content:
                self._content[document] = ' '.join(self._pipeline(doc_object=self._content[document]).text.split())
            self._output = self._content
            print('done')
            return self._output

    def _pipeline(self, doc_object):
        return self.__lemmatize__(
            self.__stop_word__(
            self.__part_of_speech__(doc_object, parts=self._attrs['nlp_processing']['part_of_speech_include'], 
            switch=self._attrs['nlp_processing']['pipeline']['pos']), 
            switch=self._attrs['nlp_processing']['pipeline']['stop']), 
            switch=self._attrs['nlp_processing']['pipeline']['lemma'])

    def __part_of_speech__(self, doc_object, parts, switch=True):
        """filter unrelated parts of speech (POS) and return required parts"""
        assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
        return self._nlp(' '.join([str(token) for token in doc_object if token.pos_ in parts])) if switch else doc_object

    def __stop_word__(self, doc_object, switch=True):
        """only remove stop words when it does not form part of phrase e.g. back pain."""
        assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
        noun_chunks = ' '.join(set([phrase.text for phrase in doc_object.noun_chunks]))
        return self._nlp(' '.join([str(token) for token in doc_object if token.is_stop is False or token.text in noun_chunks])) if switch else doc_object

    def __lemmatize__(self, doc_object, switch=True):
        assert isinstance(doc_object, spacy.tokens.doc.Doc), 'require a SpaCy document'
        return self._nlp(' '.join([str(token.lemma_) for token in doc_object])) if switch else doc_object

