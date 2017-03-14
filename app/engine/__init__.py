from app import settings
from app.map_urls import sorted_urls
from app.helpers import NHSTextMiner, NLPProcessor

TEXTMINER = settings.TEXTMINER
NHS_BASE_URL = settings.NHS_BASE_URL
NLP = settings.NLP
DATA_LOC = settings.DATA_LOC

urls = sorted_urls(base_url=NHS_BASE_URL)
web_scraper = NHSTextMiner(urls=urls, attrs=TEXTMINER, display=True)
raw_data = web_scraper.extract()

labels = {key: raw_data[key][0] for key in raw_data}
leaflets = {v: k.lower() for k, v in labels.items()}
