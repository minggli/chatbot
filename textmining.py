import pip
import spacy

__author__ = 'Ming Li'

# web scrapping module for NHS symptoms


def install(package):
    """dynamically install missing package"""
    pip.main(['install', package])

try:
    import requests
except ImportError:
    install('requests')
    import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    install('bs4')
    from bs4 import BeautifulSoup

try:
    import html5lib
except ImportError:
    install('html5lib')
    import html5lib


class NHSTextMiner(object):

    """web scrapping module using BeautifulSoup4 and Requests"""

    def __init__(self, urls, attrs, display=False):

        """urls and attrs to be supplied by main and setting."""

        assert isinstance(urls, list), 'require a list of urls'
        assert isinstance(attrs, dict), 'attributes must be a dictionary'
        # if n:
        #     assert isinstance(n, float) and n % 1 == 0 and 0 <= n <= len(urls), 'index error'

        self._urls = urls
        self._attrs = attrs
        # self._n = n
        self._count = len(urls)
        self._soups = list()
        self._display = display
        self._output = dict()

    def _get(self):

        """get all web pages and create soup objects ready for information extraction"""

        if self._display:
            print('page(s) are being downloaded...', flush=True, end='\n')

        failed_urls = list()
        for url in self._urls:
            r = requests.get(url=url)
            if self._display:
                print(r.status_code, r.url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html5lib')
                self._soups.append(soup)
            else:
                failed_urls.append(url)
                # if self._display:
                #     print('{0} downloading failed!'.format(r.url))

        for f_url in failed_urls:
            self._urls.remove(f_url)
        self._count -= len(failed_urls)



        # elif self._n:
        #
        #     n = self._n
        #     r = requests.get(url=self._urls[n])
        #     if r.status_code == 200:
        #         soup = BeautifulSoup(r.text, 'html5lib')
        #         self._soups.append(soup)

    def extract(self):

        """get all web pages and create soup objects ready for information extraction"""

        self._get()

        print('starting to extract information from websites...', flush=True, end='')

        for i, page in enumerate(self._soups):

            page_url = self._urls[i]

            subj = page.find('meta', attrs=self._attrs['subj_attributes']).get('content')
            meta = page.find('meta', attrs=self._attrs['desc_attributes']).get('content')
            article = [element.get_text(strip=True) for element in page.find_all(['p', 'li', 'meta'])]

            start_idx = int()
            end_idx = int()

            for j, value in enumerate(article):

                s1 = article[j] == self._attrs['article_attributes']['start_t_2']
                s2 = article[j + 1] == self._attrs['article_attributes']['start_t_1']
                s3 = article[j + 2] == self._attrs['article_attributes']['start_t_0']
                e1 = article[j] == self._attrs['article_attributes']['end_t_0']
                e2 = article[j + 1] == self._attrs['article_attributes']['end_t_1']
                e3 = article[j + 2] == self._attrs['article_attributes']['end_t_2']

                if s1 and s2 and s3:
                    start_idx = j + 2

                if start_idx and e1 and e2 and e3:
                    end_idx = j
                    break

            content = article[start_idx: end_idx]

            content.insert(0, subj)
            content.insert(1, meta)

            self._output[page_url] = content

        print('done')

        return self._output

    @staticmethod
    def cleanse(words, removals='''!"#$%&()*+/;<=>?@[\]^_`{|}~.,:'''):
        return [word.encode('utf-8').decode('ascii', 'ignore').translate(str.maketrans(removals, ' '*len(removals))).replace('\xa0', ' ') for word in words]

        # return [i.lower().translate(str.maketrans('', '', removals)) for i in words]

    @staticmethod
    def word_feat(words):
        return AdditiveDict([(word, True) for word in words])


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

    def __init__(self):
        """takes in raw_string or dictionary resulted from NHSTextMiner"""
        print('initiating SpaCy\'s NLP English language pipeline...', end='')
        self._nlp = spacy.load('en')
        print('done')

        self._is_string = None
        self._is_dict = None
        self._output = None
        self._doc_object = None
        self._content = None

    def process(self, content, settings={'pos': True, 'stop': True, 'lemma': True}):

        if isinstance(content, str):
            self._is_string = True
            self._doc_object = self._nlp(content)
        elif isinstance(content, dict):
            self._is_dict = True
            self._content = {key: self._nlp(' '.join(content[key])) for key in content}
        else:
            raise TypeError

        if self._is_string:
            processed = self._pipeline(doc_object=self._doc_object, settings=settings)
            self._output = ' '.join(processed.text.split())
            return self._output

        elif self._is_dict:
            for document in self._content:
                self._content[document] = ' '.join(self._pipeline(doc_object=self._content[document], settings=settings).text.split())
            self._output = self._content
            return self._output

    def _pipeline(self, doc_object, settings={'pos': True, 'stop': True, 'lemma': True}):
        return self.__lemmatize__(self.__stop_word__(
            self.__part_of_speech__(
                doc_object, switch=settings['pos']), switch=settings['stop']), switch=settings['lemma'])

    def __part_of_speech__(self, doc_object, switch=True, parts={'ADJ', 'DET', 'ADV', 'SPACE', 'CONJ', 'PRON', 'ADP', 'VERB', 'NOUN', 'PART'}):
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

