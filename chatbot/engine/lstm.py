"""
    lstm (needs modularisation and abstraction)

    document classification through recurrent neural network which models in
    sequential orders of words in their vector representation.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""
import os
import random

import tensorflow as tf
import numpy as np

from functools import wraps
from itertools import repeat
from chatbot.nlp.embedding import WordEmbedding
from sklearn.preprocessing import LabelBinarizer

from . import corpus, labels

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def resample(corpuses, labels, sample_size=1000, test_size=.2):
    """break documents into sentences and augment, and one-hot encode labels"""
    resampled_corpuses, resampled_labels = list(), list()
    for document in corpuses:
        n = corpuses.index(document)
        resampled_corpuses.append(random.choices(document, k=sample_size))
        resampled_labels.append(list(repeat(list(labels[n]), sample_size)))
    resampled_corpuses = [sent for doc in resampled_corpuses for sent in doc]
    resampled_labels = [label for doc in resampled_labels for label in doc]
    return np.array(resampled_corpuses), np.array(resampled_labels)


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
                    dynamic_pad=False,
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
STEP_SIZE = 20
N_CLASS = len(labels)

BATCH_SIZE = 50
EPOCH = 200

corpus_encoder = WordEmbedding(corpus, zero_pad=True, pad_length=STEP_SIZE)
encoded_corpus = corpus_encoder.encode()

label_encoder = LabelBinarizer().fit(labels)
encoded_labels = label_encoder.transform(labels)

features, labels = resample(encoded_corpus, encoded_labels)

embedding_matrix = corpus_encoder.vectorize()
embed_shape = embedding_matrix.shape

x = tf.placeholder(dtype=tf.int32, shape=(None, STEP_SIZE), name='feature')
y_ = tf.placeholder(dtype=tf.uint8, shape=(None, N_CLASS), name='label')

embeddings = \
        tf.get_variable(name='embeddings',
                        shape=embed_shape,
                        initializer=tf.constant_initializer(embedding_matrix),
                        trainable=False)

rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

# [BATCH_SIZE, STEP_SIZE, 300]
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
                                         inputs=rnn_inputs,
                                         dtype=tf.float32)

# outputs in shape [BATCH_SIZE, STEP_SIZE, STATE_SIZE]
# outputs = tf.transpose(a=outputs, perm=[1, 0, 2])
# last = tf.gather(params=outputs, indices=int(outputs.get_shape()[0] - 1))
# last in shape [BATCH_SIZE, STATE_SIZE]
logits = tf.matmul(final_state[1], W_softmax) + b_softmax
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

with sess:
    sent_batch, label_batch = batch_generator(*enqueue(features, labels), BATCH_SIZE)
    train(5000, x, y_, sent_batch, label_batch, train_step, accuracy, loss)
