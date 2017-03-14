import sys

from app.helpers import NLPProcessor
from app.settings import NLP

texts = str(sys.argv[1])
if not texts:
    sys.exit()

nlp = NLPProcessor(attrs=NLP)

for token in nlp._nlp(texts):
    print("{0:<15} :{1:<10} {2:>5}: {3}".format(
        token.text, token.pos_, 'post-processing', nlp.process(token.text)
        )
    )
