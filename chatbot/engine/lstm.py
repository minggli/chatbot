"""
    lstm (needs modularisation and abstraction)

    document classification through recurrent neural network which models in
    sequential orders of words in their vector representation.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""

import tensorflow as tf
import random
import numpy as np

from tqdm import tqdm
from collections import Iterable
from functools import wraps
from itertools import repeat
from chatbot.nlp.embedding import WordEmbedding
from sklearn.preprocessing import LabelBinarizer

from . import corpus, labels


def resample(corpuses, labels, sample_size=1000, test_size=.2):
    """break documents into sentences and augment, and one-hot encode labels"""
    resampled_corpuses, resampled_labels = list(), list()
    for document in tqdm(corpuses, miniters=1):
        n = corpuses.index(document)
        resampled_corpuses.append(random.choices(document, k=sample_size))
        resampled_labels.append(list(repeat(list(labels[n]), sample_size)))
    resampled_corpuses = [sent for doc in resampled_corpuses for sent in doc]
    resampled_labels = [label for doc in resampled_labels for label in doc]
    return np.array(resampled_corpuses), np.array(resampled_labels)


def encode_corpus(iterable, word_id_dict):
    """recursively map word with id or 0 if not in vocabulary"""
    assert isinstance(iterable, Iterable), isinstance(word_id_dict, dict)
    converted = list()
    for subset in iterable:
        if isinstance(subset, list):
            converted.append(encode_corpus(subset, word_id_dict))
        elif isinstance(subset, str):
            converted.append(word_id_dict.get(subset, 0))
    return converted


def zero_pad(iterable, length=None):
    """pad shorter sequences with 0 (as if out of vocabulary), temporary until
    figure out dynamic padding (e.g. train.batch)"""
    length = length or max(len(x) for x in iterable)

    for subset in iterable:
        if not any(not isinstance(x, (int, str)) for x in subset):
            subset.extend([0] * (length - len(subset)))
            subset = subset[:length]
        elif not any(not isinstance(x, list) for x in subset):
            zero_pad(subset, length)
    return iterable


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
    sent = tf.convert_to_tensor(sent_encoded, dtype=tf.int32)
    label = tf.convert_to_tensor(label_encoded, dtype=tf.uint8)
    input_queue = tf.train.slice_input_producer(
                                tensor_list=[sent, label],
                                num_epochs=num_epochs,
                                shuffle=shuffle)
    return input_queue


def batch_generator(sent_queue, label_queue, batch_size=None, threads=4):
    return tf.train.batch(
                    tensors=[sent_queue, label_queue],
                    batch_size=batch_size,
                    # critical for varying sequence length, pad with 0 or ' '
                    dynamic_pad=True,
                    num_threads=threads,
                    capacity=1e3,
                    allow_smaller_final_batch=True)


@multithreading
def train(n, x, y_, sent_batch, label_batch, optimiser, metric, loss):
    for global_step in range(n):
        sent, label = sess.run(fetches=[sent_batch, label_batch])

        _, train_accuracy, train_loss = \
            sess.run(fetches=[optimiser, metric, loss],
                     feed_dict={x: sent, y_: label})
        print("step {0} of {3}, train accuracy: {1:.4f}"
              " log loss: {2:.4f}".format(global_step, train_accuracy,
                                          train_loss, n))


STATE_SIZE = 24
STEP_SIZE = 200
N_CLASS = len(labels)

BATCH_SIZE = 50
EPOCH = 200

corpus_encoder = WordEmbedding(corpus)
embedding_matrix = corpus_encoder.vectorize()
embed_shape = embedding_matrix.shape
encoded_corpus = corpus_encoder.encode()

print(encoded_corpus[0][:5])
# v = Vectorizer()
# tokens = [[[word.text for word in v(sents)] for sents in document]
#           for document in corpus]
# vocab = islice(zip(*Counter(chain(*chain(*tokens))).most_common()), 1)
# word_to_ids = {word: ids for ids, word in enumerate(*vocab, start=1)}
#
# embedding_matrix = v.fit(word_to_ids).transform()
#
# encoded_corpus = zero_pad(encode_corpus(tokens, word_to_ids), length=STEP_SIZE)

label_encoder = LabelBinarizer().fit(labels)
encoded_labels = label_encoder.transform(labels)

features, labels = resample(encoded_corpus, encoded_labels)

x = tf.placeholder(dtype=tf.int32, shape=(None, STEP_SIZE), name='feature')
y_ = tf.placeholder(dtype=tf.uint8, shape=(None, N_CLASS), name='label')
v_ = tf.placeholder(dtype=tf.float32, shape=embed_shape, name='vector')

embeddings = tf.get_variable(name='W',
                             shape=embed_shape,
                             initializer=tf.constant_initializer(0.0),
                             trainable=False)

word_vectors = tf.nn.embedding_lookup(embeddings, x)

W_softmax = tf.get_variable(
                    name='W_yh',
                    shape=[STATE_SIZE, N_CLASS],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
b_softmax = tf.get_variable(
                    name='b_y',
                    shape=[N_CLASS],
                    initializer=tf.constant_initializer(0.0))

cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)
initial_state = cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                         inputs=word_vectors,
                                         initial_state=initial_state)
# [200, 24]
logits = tf.matmul(outputs[-1], W_softmax) + b_softmax
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
init_embedd = embeddings.assign(v_)

sess = tf.Session()
sess.run(fetches=[init_embedd, init], feed_dict={v_: embedding_matrix})

with sess:
    sent_batch, label_batch = batch_generator(*enqueue(features, labels), BATCH_SIZE)
    train(1000, x, y_, sent_batch, label_batch, train_step, accuracy, loss)
