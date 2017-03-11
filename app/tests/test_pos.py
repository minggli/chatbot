import sys

from ..helpers import NLPProcessor
from ..settings import NLP

texts = sys.argv[1]
if not texts:
    sys.exit()

nlp = NLPProcessor(attrs=NLP)

for token in nlp._nlp(texts):
    print("{0:<15} :{1:<10} {2:>5}: {3}".format(token.text, token.pos_, 'post-processing', nlp.process(token.text)))
