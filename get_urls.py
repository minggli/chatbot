import pip
import string
import pickle
import os
import re


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

index = list(string.ascii_uppercase) + ['0-9']
Base_Url = 'http://www.nhs.uk'
complete_urls = list()


def extract_index_pages(file='data/index_pages.pkl'):

    print('constructing website skeleton...', end='\n')
    index_urls = [Base_Url + '/Conditions/Pages/BodyMap.aspx?Index={}'.format(i) for i in index]

    bs4_objects = list()

    if not os.path.exists(file):
        try:
            os.mkdir(file[:5])
        except FileExistsError:
            pass

        for url in index_urls:
            r = requests.get(url=url)
            print(r.status_code, r.url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html5lib')
                bs4_objects.append(soup)
        with open(file, 'wb') as filename:
            pickle.dump(bs4_objects, filename)
    else:
        with open(file, 'rb') as f:
            bs4_objects = pickle.load(f)

    return bs4_objects


def extract_hyperlinks(page, base_url=Base_Url, regex='/[Cc]onditions/.*'):

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
    urls = list(set(map(lambda x: base_url + x, urls)))
    return urls

pages = extract_index_pages()

for page in pages:
    complete_urls += extract_hyperlinks(page)

web_pages = {k: v for k, v in enumerate(set(complete_urls))}
