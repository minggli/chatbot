import requests
from bs4 import BeautifulSoup

__author__ = 'Ming Li'

# web scrapping module for NHS symptoms


class NHSTextMining(object):

    """web scrapping module using BeautifulSoup4 and Requests"""

    def __init__(self, urls, attrs, n=None, display=False):

        """urls and attrs to be supplied by main and setting."""

        assert isinstance(urls, list), 'require a list of urls'
        assert isinstance(attrs, dict), 'attributes must be a dictionary'
        if n:
            assert isinstance(n, float) and n % 1 == 0 and 0 <= n <= len(urls), 'index error'

        self._urls = urls
        self._attrs = attrs
        self._n = n
        self._count = len(urls)
        self._soups = list()
        self._display = display
        self._output = dict()

    def _get(self):

        """get all web pages and create soup objects ready for information extraction"""

        if self._display:
            print('page(s) are being downloaded...', flush=True, end='')

        if not self._n:

            for i in range(self._count):
                r = requests.get(url=self._urls[i])
                soup = BeautifulSoup(r.text, 'html5lib')
                self._soups.append(soup)

        elif self._n:

            n = self._n
            r = requests.get(url=self._urls[n])
            soup = BeautifulSoup(r.text, 'lxml')

            self._soups.append(soup)

        if self._display:
            print('done')

    def extract(self):

        """get all web pages and create soup objects ready for information extraction"""

        self._get()

        print('starting to extract information from websites...', flush=True, end='')

        for i, page in enumerate(self._soups):

            page_url = self._urls[i]

            subj = page.find('meta', attrs=self._attrs['subj_attributes']).get('content')
            meta = page.find('meta', attrs=self._attrs['desc_attributes']).get('content')
            article = [element.get_text() for element in page.find_all(['p', 'li', 'meta'])]

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
        return [i.lower().translate(str.maketrans('', '', removals)).replace('\xa0', ' ') for i in words]

    @staticmethod
    def word_feat(words):
        return dict([(word, True) for word in words])
