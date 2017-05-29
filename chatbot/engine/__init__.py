from chatbot import settings as sett
from chatbot.ie import extracted_urls, TextMiner
from chatbot.helpers import NLPProcessor

TEXTMINER, NHS_BASE_URL, NLP = sett.TEXTMINER, sett.NHS_BASE_URL, sett.NLP

urls = extracted_urls(base_url=NHS_BASE_URL)
web_scraper = TextMiner(urls=urls, attrs=TEXTMINER, display=True)
web_scraper.extract()

raw_data = web_scraper.jsonify()
corpus = [json['doc'] for json in raw_data]
labels = [json['label'] for json in raw_data]
leaflets = {json['label']: json['url'].lower() for json in raw_data}
