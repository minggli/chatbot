
from chatbot import settings as s
from chatbot.ie import extracted_urls, TextMiner
from chatbot.nlp.embedding import WordVectorizer


WEB_METAKEY, BASE_URL = s.WEB_METAKEY, s.BASE_URL

urls = extracted_urls(base_url=BASE_URL)
web_scraper = TextMiner(urls=urls, attrs=WEB_METAKEY, display=True)

raw_data = web_scraper.extract().jsonify()
corpus = [json['doc'] for json in raw_data]

v = WordVectorizer()
corpus = [token.text for doc in corpus for chunk in doc for token in v(chunk)]
vectors = v.fit(corpus).transform()

unrepresented_words = list()
for k, w in enumerate(corpus):
    if vectors[k].all() == 0:
        unrepresented_words.append(w)

unrepresented_words.sort(key=lambda x: len(x), reverse=True)

for w in unrepresented_words:
    print(w)
    if len(w) < 5:
        break

print('{0:.4f}%'.format(len(unrepresented_words) / len(corpus) * 100))
