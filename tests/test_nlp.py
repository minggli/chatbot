import sys

from chatbot.helpers import NLPProcessor
from chatbot.settings import NLP

try:
    texts = str(sys.argv[1])
except IndexError:
    raise RuntimeError('there is no texts in command line.')

nlp = NLPProcessor(attrs=NLP)

for token in nlp._nlp(texts):
    print("{0:<15} :{1:<10} {2:>5}: {3}".format(
        token.text, token.pos_, 'post-processing', nlp.process(token.text)
        )
    )
