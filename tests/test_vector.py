
from chatbot import settings as s
from chatbot.ie import extracted_urls, TextMiner
from chatbot.nlp.embedding import WordVectorizer


TEXTMINER, NHS_BASE_URL = s.TEXTMINER, s.NHS_BASE_URL

urls = extracted_urls(base_url=NHS_BASE_URL)
web_scraper = TextMiner(urls=urls, attrs=TEXTMINER, display=True)

raw_data = web_scraper.extract().jsonify()
corpus = [json['doc'] for json in raw_data]

corpus = [word for doc in corpus for text in doc for word in text.split()]
vectors = WordVectorizer().fit(corpus).transform()

unrepresented_words = list()
for k, w in enumerate(corpus):
    if vectors[k].all() == 0:
        unrepresented_words.append(w)

print('{0:.4f}% of {1} unrepresented'.format(
      (len(unrepresented_words) / len(corpus) * 100), len(corpus))
