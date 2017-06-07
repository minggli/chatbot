"""
    lstm

    document classification through recurrent neural network which models in
    sequential orders of words in their vector representation.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from collections import Counter, Iterable
from itertools import repeat, chain, islice
from chatbot.nlp.embedding import VectorLookup
from sklearn.preprocessing import LabelBinarizer

from . import corpus, labels

STATE_SIZE = 24
STEP_SIZE = 50
N_CLASS = len(labels)

# for testing purpose, fixing random seed for determinisim
tf.set_random_seed(1)


def pipeline(corpuses, labels, sample_size=1000, test_size=.2):
    """break documents into sentences and augment, and one-hot encode labels"""
    assert isinstance(corpuses, Iterable), 'corpuses not iterable.'
    feat, label = list(), list()
    label_encoder = LabelBinarizer().fit(labels)
    for document in tqdm(corpuses, miniters=1):
        n = corpuses.index(document)
        feat.append(np.random.choice(document, sample_size, replace=True))
        label.append(list(repeat(labels[n], sample_size)))
    feat, label = np.ravel(feat), np.ravel(label)
    return feat, label_encoder.transform(label)


def encode_words(iterable, word_id_dict):
    """recursively map word with id or 0 if not found"""
    assert isinstance(iterable, Iterable)
    assert isinstance(word_id_dict, dict)
    converted = list()
    for subset in iterable:
        if isinstance(subset, list):
            converted.append(encode_words(subset, word_id_dict))
        elif isinstance(subset, str):
            converted.append(word_id_dict.get(subset, 0))
    return converted


v = VectorLookup()
tokens = [[[word.text for word in v(sents)] for sents in document]
          for document in corpus]
vocab = islice(zip(*Counter(chain(*chain(*tokens))).most_common()), 1)
word_to_ids = {word: ids for ids, word in enumerate(*vocab, start=1)}
embedd = v.fit(word_to_ids).transform()
encoded_corpus = encode_words(tokens, word_to_ids)

feature, labels = pipeline(encoded_corpus, labels)

x = tf.placeholder(dtype=tf.float32, shape=(None, STEP_SIZE))
y_ = tf.placeholder(dtype=tf.float32, shape=(None, N_CLASS))
e = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False, name="W")
embedding_init = W.assign(e)

W_softmax = tf.get_variable(
                    name='W_yh',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
b_softmax = tf.get_variable(
                    name='b_y',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0))

cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)

outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                         inputs=x,
                                         initial_state=cell.zero_state)

logits = tf.matmul(final_state, W_softmax) + b_softmax
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer().minimize(loss)

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
