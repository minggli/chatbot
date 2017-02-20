from bs4 import BeautifulSoup
from settings import DATA_LOC
import requests
import html5lib
import string
import pickle
import os, sys
sys.setrecursionlimit(30000)
import re


def extract_index_pages(base_url):

    index = list(string.ascii_uppercase) + ['0-9']

    index_urls = [base_url + '/Conditions/Pages/BodyMap.aspx?Index={}'.format(i) for i in index]

    bs4_objects = list()

    if not os.path.exists(DATA_LOC + 'index_pages.pkl'):
        try:
            os.mkdir(DATA_LOC)
        except FileExistsError:
            pass

        print('constructing {} skeleton of symptom pages...'.format(base_url), end='\n')

        for url in index_urls:
            r = requests.get(url=url)
            print(r.status_code, r.url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html5lib')
                bs4_objects.append(soup)
        with open(DATA_LOC + 'index_pages.pkl', 'wb') as filename:
            pickle.dump(bs4_objects, filename)
    else:
        with open(DATA_LOC + 'index_pages.pkl', 'rb') as f:
            bs4_objects = pickle.load(f)

    return bs4_objects


def extract_hyperlinks(page, base_url, regex='/[Cc]onditions/.*'):

    urls = list()
    for element in page.find_all(['a']):
        url = re.findall(pattern=regex, string=element.get('href'))
        if len(url) != 0:
            url = str(url[0]).lower()
            end_index = url.find('/pages/')
            if end_index > 0:
                url = url[:end_index]
            if len(url) > 11:
                urls.append(url)
    urls = list(set(map(lambda x: base_url + x.strip(), urls)))
    return urls


def sorted_urls(base_url):
    complete_urls = [extract_hyperlinks(page=p, base_url=base_url)
    for p in extract_index_pages(base_url=base_url)][0]
    sorted_urls = sorted(list(set(complete_urls)))
    return sorted_urls