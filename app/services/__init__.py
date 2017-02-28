from .. import settings
from ..map_urls import sorted_urls
from ..helpers import NHSTextMiner

TEXTMINER = settings.TEXTMINER
NHS_BASE_URL = settings.NHS_BASE_URL
NLP = settings.NLP
API_BASE_URL = settings.API_BASE_URL

urls = sorted_urls(base_url=NHS_BASE_URL)
web_scraper = NHSTextMiner(urls=urls, attrs=TEXTMINER, display=True)
raw_data = web_scraper.extract()

labels = {key: raw_data[key][0] for key in raw_data}
mapping = {v: k for k, v in labels.items()}
