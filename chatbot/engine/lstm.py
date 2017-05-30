"""
    lstm

    document classification through recurrent neural network which models in
    sequential orders of words in their vector representation.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""

import tensorflow as tf

from chatbot.nlp.embedding import WordVectorizer
from . import corpus, labels


v = WordVectorizer()
corpus = [token.text for doc in corpus for chunk in doc for token in v(chunk)]
vectors = v.fit(corpus).transform()

unrepresented_words = list()
for k, w in enumerate(corpus):
    if vectors[k].all() == 0:
        unrepresented_words.append(w)

unrepresented_words.sort(key=lambda x: len(x), reverse=True)

for w in unrepresented_words:
    print(w)
    if len(w) < 5:
        break

print('{0:.4f}%'.format(len(unrepresented_words) / len(corpus) * 100))



# corpus.sort(key=lambda x: len(x), reverse=True)
#
# for w in corpus:
#     print(w)
