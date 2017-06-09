"""
    lstm (needs modularisation and abstraction)

    document classification through recurrent neural network which models in
    sequential orders of words in their vector representation.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from collections import Counter, Iterable
from functools import wraps
from itertools import repeat, chain, islice
from chatbot.nlp.embedding import VectorLookup
from sklearn.preprocessing import LabelBinarizer

from . import corpus, labels


def resample(corpuses, labels, sample_size=1000, test_size=.2):
    """break documents into sentences and augment, and one-hot encode labels"""
    assert isinstance(corpuses, Iterable), 'corpuses not iterable.'
    resampled_corpuses, resampled_labels = list(), list()
    for document in tqdm(corpuses, miniters=1):
        n = corpuses.index(document)
        resampled_corpuses.append(np.random.choice(document, sample_size,
                                                   replace=True))
        resampled_labels.append(list(repeat(labels[n], sample_size)))
    resampled_corpuses, resampled_labels = \
        np.ravel(resampled_corpuses), np.ravel(resampled_labels)
    return resampled_corpuses, resampled_labels


def encode_words(iterable, word_id_dict):
    """recursively map word with id or 0 if not in vocabulary"""
    assert isinstance(iterable, Iterable), isinstance(word_id_dict, dict)
    converted = list()
    for subset in iterable:
        if isinstance(subset, list):
            converted.append(encode_words(subset, word_id_dict))
        elif isinstance(subset, str):
            converted.append(word_id_dict.get(subset, 0))
    return converted


def multithreading(func):
    """decorator using tensorflow threading ability."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        func_output = func(*args, **kwargs)
        try:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=5)
        except (tf.errors.CancelledError, RuntimeError) as e:
            pass
        return func_output
    return wrapper


def enqueue(sent_encoded, label_encoded, num_epochs=None, shuffle=True):
    """returns an Ops Tensor with queued sentence and label pair"""
    sent = tf.convert_to_tensor(sent_encoded, dtype=tf.uint8)
    label = tf.convert_to_tensor(label_encoded, dtype=tf.uint8)
    input_queue = tf.train.slice_input_producer(
                                tensor_list=[sent, label],
                                num_epochs=num_epochs,
                                shuffle=shuffle)
    return input_queue


def batch_generator(sent_encoded, label_encoded, batch_size=None, threads=4):
    return tf.train.batch(
                    tensors=[sent_encoded, label_encoded],
                    batch_size=batch_size,
                    # critical for varying sequence length, pad with 0 or ' '
                    dynamic_pad=True,
                    num_threads=threads,
                    capacity=1e-3,
                    allow_smaller_final_batch=True)


STATE_SIZE = 24
STEP_SIZE = 200
N_CLASS = len(labels)

BATCH_SIZE = 50
EPOCH = 200

v = VectorLookup()
tokens = [[[word.text for word in v(sents)] for sents in document]
          for document in corpus]
vocab = islice(zip(*Counter(chain(*chain(*tokens))).most_common()), 1)
word_to_ids = {word: ids for ids, word in enumerate(*vocab, start=1)}
encoded_corpus = encode_words(tokens, word_to_ids)

label_encoder = LabelBinarizer().fit(labels)
encoded_labels = label_encoder.transform(labels)

features, labels = resample(encoded_corpus, encoded_labels)
sent_batch, label_batch = batch_generator(enqueue(features, labels, EPOCH))
embedding_matrix = v.fit(word_to_ids).transform()
embed_shape = embedding_matrix.shape

x = tf.placeholder(dtype=tf.int32, shape=(None, STEP_SIZE), name='feature')
y_ = tf.placeholder(dtype=tf.uint8, shape=(None, N_CLASS), name='one-hot')
v_ = tf.placeholder(dtype=tf.float32, shape=embed_shape, name='vector')

W = tf.get_variable(name='W',
                    shape=embed_shape,
                    initializer=tf.constant_initializer(0.0),
                    trainable=False)

vectorized_x = tf.nn.embedding_lookup(W, x)

W_softmax = tf.get_variable(
                    name='W_yh',
                    shape=[STATE_SIZE, N_CLASS],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
b_softmax = tf.get_variable(
                    name='b_y',
                    shape=[N_CLASS],
                    initializer=tf.constant_initializer(0.0))

cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)

outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                         inputs=vectorized_x,
                                         dtype=tf.float32)

logits = tf.matmul(outputs[-1], W_softmax) + b_softmax
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
init_embedd = W.assign(v_)

sess = tf.Session()
sess.run(fetches=[init_embedd, init], feed_dict={v_: embedding_matrix})

sent, label = sess.run(sent_batch, label_batch)
print(sent)
sess.run(train_step, {x: sent, y_: label})
