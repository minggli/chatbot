"""
    lstm

    document classification through recurrent neural network which models in
    sequential orders of words in their vector representation.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""


from chatbot.nlp.embedding import WordVectorizer
from . import corpus


last_doc = ' '.join(corpus[-1]).split()
test = ['the', 'brown', 'fox', 'jumps']

vectors = WordVectorizer().fit(test).transform()

for k, w in enumerate(test):
    print(w)
    print(vectors[k])
