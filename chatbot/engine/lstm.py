"""
    lstm

    document classification through recurrent neural network which models in
    sequential orders of words in their vector representation.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""

from . import corpus, labels


last = corpus[-1]

arti = [texts for texts in last]
for i in arti:
    print(i)
