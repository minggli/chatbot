"""
    ie

    informatio extraction module provides a mapping of useful urls that can be
    used as training data.
"""

import os
import sys

import re
import string
import pickle
import requests

from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from multiprocessing import Pool

from .settings import DATA_LOC, EN_CONTRACTIONS

sys.setrecursionlimit(30000)
# # TODO find a better way to cache BeautifulSoup objects


def extract_index_pages(base_url):
    """obtain BodyMap A-Z web pages to extract skeleton of symptom pages."""
    index = list(string.ascii_uppercase) + ['0-9']
    index_urls = [base_url + '/Conditions/Pages/BodyMap.aspx?Index={}'.format(
                  i) for i in index]
    bs4_objects = list()
    if not os.path.exists(DATA_LOC + 'index_pages.pkl'):
        try:
            os.mkdir(DATA_LOC)
        except FileExistsError:
            pass
        print('constructing {} skeleton of symptom pages...'.format(
            base_url), end='\n')
        for url in index_urls:
            r = requests.get(url=url)
            print(r.status_code, r.url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html5lib')
                bs4_objects.append(soup)
        with open(DATA_LOC + 'index_pages.pkl', 'wb') as f:
            pickle.dump(bs4_objects, f)
    else:
        with open(DATA_LOC + 'index_pages.pkl', 'rb') as f:
            bs4_objects = pickle.load(f)
    return bs4_objects


def extract_hyperlinks(page, base_url, regex='/[Cc]onditions/.*'):
    """extract hyperlinks from pages, if they are symptom-related leaflets."""
    urls = list()
    for element in page.find_all(['a']):
        url = re.findall(pattern=regex, string=element.get('href'))
        if url:
            url = str(url[0]).lower()
            end_index = url.find('/pages/')
            if end_index > 0:
                url = url[:end_index]
            if len(url) > 11:
                urls.append(url)
    urls = list(set(map(lambda x: base_url + x.strip(), urls)))
    return urls


def extracted_urls(base_url):
    """produce a sorted list of unique urls alphabetically."""
    s = requests.Session()
    s.mount(base_url, HTTPAdapter(max_retries=5))

    nested_list = [extract_hyperlinks(page=p, base_url=base_url)
                   for p in extract_index_pages(base_url=base_url)]
    unravelled_list = [url for l in nested_list for url in l]
    return sorted(list(set(unravelled_list)))


class TextMiner:
    """web scrapping module using BeautifulSoup4 and Requests"""

    def __init__(self, urls, attrs, threads=4, display=False):

        self._urls = urls
        self._attrs = None
        self._display = display
        self._count = len(urls)
        self._failed_urls = list()
        self._soups = list()
        self._output = dict()
        self._threads = threads

        self.__attrs__ = attrs

    @property
    def __attrs__(self):
        return self._attrs

    @__attrs__.setter
    def __attrs__(self, values):
        if not all([isinstance(values, dict), len(values) >= 3]):
            raise TypeError('attributes must be a dictionary and contain'
                            ' desc_attributes, subj_attribtues, and '
                            'article_attributes.')
        else:
            self._attrs = values

    def _get(self, url):
        """get all web pages and create soup objects ready for extraction
        """

        try:
            r = requests.get(url=url)
        except requests.exceptions.ConnectionError:
            raise RuntimeError('There is no active Internet connection.')

        if self._display:
            print(r.status_code, r.url)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            return tuple((soup, None))
        else:
            return tuple((None, url))

    def _mp_get(self):
        """multiprocess _get function for performance."""
        if self._display:
            print('{0} pages are being downloaded...'.format(
                len(self._urls)), flush=True, end='\n')

        with Pool(self._threads) as p:
            merged_output = p.map(self._get, self._urls)

        self._failed_urls = [pair[1] for pair in merged_output if pair[
            0] is None and pair[1] is not None]
        self._soups = [pair[0] for pair in merged_output if pair[
            0] is not None and pair[1] is None]

        for f_url in self._failed_urls:
            self._urls.remove(f_url)
            self._count -= 1

    def extract(self):
        """get all web pages and create soup objects ready for extraction"""

        if not os.path.exists(DATA_LOC + 'symptoms.pkl'):

            self._mp_get()

            print('starting to extract information from websites...',
                  flush=True, end='')

            self._failed_urls.clear()

            for i, page_url in enumerate(self._urls):
                page = self._soups[i]
                try:
                    subj = page.find('meta', attrs=self._attrs[
                        'subj_attributes']).get('content')
                    meta = page.find('meta', attrs=self._attrs[
                        'desc_attributes']).get('content')
                    aricl = [i.get_text()
                             for i in page.find_all(['p', 'li'])
                             if not i.find(class_='hidden')]
                except AttributeError:
                    self._failed_urls.append(page_url)
                    continue
                for i in aricl:
                    print(i)

                start_idx = int()
                end_idx = int()

                for j, value in enumerate(aricl):
                    # using 3 keys each end to identify range of main article
                    try:
                        s1 = aricl[j] == \
                            self._attrs['article_attributes']['start_t_2']
                        s2 = aricl[j + 1] == \
                            self._attrs['article_attributes']['start_t_1']
                        s3 = aricl[j + 2] == \
                            self._attrs['article_attributes']['start_t_0']
                        e1 = aricl[j] == \
                            self._attrs['article_attributes']['end_t_0']
                        e2 = aricl[j + 1] == \
                            self._attrs['article_attributes']['end_t_1']
                        e3 = aricl[j + 2] == \
                            self._attrs['article_attributes']['end_t_2']
                    except IndexError:
                        self._failed_urls.append(page_url)
                        break

                    if s1 and s2 and s3:
                        start_idx = j + 2

                    if start_idx and e1 and e2 and e3:
                        end_idx = j
                        break

                content = aricl[start_idx: end_idx]

                if len(content) < 5:
                    self._failed_urls.append(page_url)
                    continue

                content.insert(0, subj)
                content.insert(1, meta)

                content = self.cleanse_content(content)

                self._output[page_url] = content

            for f_url in list(set(self._failed_urls)):
                self._urls.remove(f_url)
                self._count -= 1

            print('done. {} of {} failed to be extracted.'.format(
                len(set(self._failed_urls)), len(self._soups)), flush=True)

            with open(DATA_LOC + 'symptoms.pkl', 'wb') as f:
                pickle.dump(obj=self._output, file=f)
        else:
            with open(DATA_LOC + 'symptoms.pkl', 'rb') as f:
                self._output = pickle.load(f)

        return self

    def cleanse_content(self, content):
        cleansed_content = list()
        trans = str.maketrans("–’£ ", "-'  ")
        for text in content:
            if not content.index(text):
                text = text.replace(' - NHS Choices', '')
            text = text.translate(trans)
            text = self.__class__.remove_email(text)
            text = self.__class__.split_contraction(text)
            cleansed_content.append(text)
        return cleansed_content

    def jsonify(self):
        return [{"url": key,
                 "label": self._output[key][0],
                 "doc": self._output[key][1:]} for key in self._output]

    @staticmethod
    def split_contraction(texts, lib=EN_CONTRACTIONS):
        """split you'll to you will and other similar cases"""
        regex = re.compile(pattern='({0})'.format('|'.join(lib.keys())),
                           flags=re.IGNORECASE)
        return regex.sub(lambda x: lib[str.lower(x.group(0))], texts)

    @staticmethod
    def remove_email(texts):
        """detect email address substrings and remove"""
        regex = re.compile(pattern=r'[\w\.-]+@[\w\.-]+', flags=re.IGNORECASE)
        return regex.sub(lambda x: '', texts)
