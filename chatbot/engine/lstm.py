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
from chatbot.nlp.embedding import WordEmbedding
from sklearn import model_selection, preprocessing

from . import corpus, labels

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def flatten_split_resample(encoded_corpuses, encoded_labels,
                           valid_ratio=.2,
                           sample_size=1000):
    """break documents into sentences and augment, and one-hot encode labels"""

    def resample(docs, labels, sample_size=sample_size):
        unique_labels = np.unique(labels)
        indice_array = list()
        for label in unique_labels:
            indice_label = np.where(labels == label)[0].tolist()
            indice_array.extend(random.choices(indice_label, k=sample_size))
        return (docs[indice_array], labels[indice_array])

    flattened_docs = list()
    flattened_labels = list()
    for d, l in zip(encoded_corpuses, encoded_labels):
        for sent in d:
            flattened_docs.append(sent)
            flattened_labels.append(l)

    ravelled_corpus = np.array(flattened_docs)
    ravelled_labels = np.array(flattened_labels)

    X_train, X_test, y_train, y_test = \
        model_selection.train_test_split(ravelled_corpus, ravelled_labels,
                                         test_size=valid_ratio,
                                         stratify=ravelled_labels)

    return resample(X_train, y_train), resample(X_test, y_test)


def size(sequence):
    """produces length vector in shape [batch_size, length] for each padded
    sequence of its true length excluding zero vectors.
    sequence of vectors comes in shape of [batch_size, STEP_SIZE, dimensions]
    """
    flag = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    # [BATCH_SIZE, STEP_SIZE] with 0 or 1
    length = tf.reduce_sum(flag, axis=1)
    # [BATCH_SIZE] with number of 1s used as true lengths
    return tf.cast(length, tf.int32)


def last_outputs(outputs, length):
    """zero padded sequences will produce zero vectors after sequence_length
    required using dynamic unrolling dynamic_rnn API. this function locates
    the last non-zero output for calculating logits
    """
    # outputs in shape [BATCH_SIZE, STEP_SIZE, STATE_SIZE]
    # length in shape [BATCH_SIZE]
    b, t, st = tf.unstack(tf.shape(outputs))
    flat_index = tf.range(b) * t + (length - 1)
    # flat_index 1-D tensor in shape [BATCH_SIZE]
    flat_outputs = tf.reshape(outputs, [-1, st])
    # outputs flattened to [BATCH_SIZE * STEP_SIZE, STATE_SIZE]
    last_outputs = tf.gather(flat_outputs, flat_index)
    return last_outputs


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
    return tf.train.shuffle_batch(
                    tensors=[sent_queue, label_queue],
                    batch_size=batch_size,
                    num_threads=threads,
                    capacity=1e3,
                    min_after_dequeue=batch_size,
                    allow_smaller_final_batch=True)


@multithreading
def train(n, x, y_, train_sent_batch, train_label_batch, valid_sent_batch,
          valid_label_batch, keep_prob, optimiser, metric, loss):

    for global_step in range(n):
        train_sent, train_label = \
                    sess.run(fetches=[train_sent_batch, train_label_batch])
        _, train_accuracy, train_loss = \
            sess.run(fetches=[optimiser, metric, loss],
                     feed_dict={x: train_sent, y_: train_label, keep_prob: 1})

        print("step {0} of {3}, train accuracy: {1:.4f} log loss: {2:.4f}"
              .format(global_step, train_accuracy, train_loss, n))

        if global_step and global_step % 10 == 0:
            valid_sent, valid_label = \
                    sess.run(fetches=[valid_sent_batch, valid_label_batch])
            valid_accuracy, valid_loss = sess.run(
                    fetches=[metric, loss],
                    feed_dict={x: valid_sent, y_: valid_label, keep_prob: 1})

            print("step {0} of {3}, valid accuracy: {1:.4f} log loss: {2:.4f}"
                  .format(global_step, valid_accuracy, valid_loss, n))


N_DOC = 999
corpus = corpus[:N_DOC]
labels = labels[:N_DOC]

STATE_SIZE = 24
STEP_SIZE = 200
N_CLASS = len(labels)
BATCH_SIZE = 50
EPOCH = 200

corpus_encoder = WordEmbedding(raw_corpus=corpus, pad_length=STEP_SIZE, n=None)
encoded_corpus = corpus_encoder.encode()

label_encoder = preprocessing.LabelBinarizer().fit(labels)
encoded_labels = label_encoder.transform(labels)

embedding_matrix = corpus_encoder.vectorize()
embed_shape = embedding_matrix.shape

x = tf.placeholder(dtype=tf.int32, shape=(None, STEP_SIZE), name='feature')
y_ = tf.placeholder(dtype=tf.uint8, shape=(None, N_CLASS), name='label')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_rate')

embeddings = tf.get_variable(
                        name='embeddings',
                        shape=embed_shape,
                        initializer=tf.constant_initializer(embedding_matrix),
                        trainable=False)

rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
# [BATCH_SIZE, STEP_SIZE, DIMENSIONS]

with tf.variable_scope('softmax'):
    W_softmax = tf.get_variable(
                    name='W',
                    shape=[STATE_SIZE, N_CLASS],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_softmax = tf.get_variable(
                    name='b',
                    shape=[N_CLASS],
                    initializer=tf.constant_initializer(0.0))

cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)
cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                                     input_keep_prob=keep_prob,
                                     output_keep_prob=keep_prob)
sent_length = size(rnn_inputs)
outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                         inputs=rnn_inputs,
                                         sequence_length=sent_length,
                                         dtype=tf.float32)

# outputs in shape [BATCH_SIZE, STEP_SIZE, STATE_SIZE]
last = last_outputs(outputs, sent_length)
logits = tf.matmul(last, W_softmax) + b_softmax
# logits in shape [BATCH_SIZE, N_CLASS]
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_)
# softmax cross entropy in shape [BATCH_SIZE, ]
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

with sess, tf.device('/cpu:0'):
    data = flatten_split_resample(encoded_corpus, encoded_labels,
                                  valid_ratio=.2,
                                  sample_size=3000)
    train_sent_batch, train_label_batch = \
        batch_generator(*enqueue(*data[0]), BATCH_SIZE)
    valid_sent_batch, valid_label_batch = \
        batch_generator(*enqueue(*data[1]), BATCH_SIZE)
    train(5000, x, y_, train_sent_batch, train_label_batch, valid_sent_batch,
          valid_label_batch, keep_prob, train_step, accuracy, loss)
