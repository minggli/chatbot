from chatbot import settings as s
from chatbot.ie import extracted_urls, TextMiner
from chatbot.nlp.sparse import NLPProcessor

TEXTMINER, NHS_BASE_URL, NLP = s.TEXTMINER, s.NHS_BASE_URL, s.NLP

urls = extracted_urls(base_url=NHS_BASE_URL)
web_scraper = TextMiner(urls=urls, attrs=TEXTMINER, display=True)

raw_data = web_scraper.extract().jsonify()
corpus = [json['doc'] for json in raw_data][:20]
labels = [json['label'] for json in raw_data][:20]
leaflets = {json['label']: json['url'].lower() for json in raw_data}
