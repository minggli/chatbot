from chatbot import settings
from chatbot.ie import extracted_urls, TextMiner

TEXTMINER, NHS_BASE_URL = settings.TEXTMINER, settings.NHS_BASE_URL

urls = extracted_urls(base_url=NHS_BASE_URL)
web_scraper = TextMiner(urls=urls, attrs=TEXTMINER, display=True)

raw_data = web_scraper.extract().jsonify()
corpus = [json['doc'] for json in raw_data]
labels = [json['label'] for json in raw_data]
leaflets = {json['label']: json['url'].lower() for json in raw_data}
