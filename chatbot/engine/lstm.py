"""
    lstm

    document classification through recurrent neural network which models in
    sequential orders of words in their representational forms.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""

from . import TextMiner, NLPProcessor, NLP, raw_data, labels


one = raw_data

arti = [texts for texts in list(one.values())[3]]
for i in arti:
    print(i.encode('latin'))
