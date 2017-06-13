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
from sklearn import model_selection, preprocessing

from . import corpus, labels

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def resample(corpuses, labels, sample_size=1000):
    """break documents into sentences and augment, and one-hot encode labels"""
    resampled_corpuses, resampled_labels = list(), list()
    for document in corpuses:
        n = corpuses.index(document)
        resampled_corpuses.append(random.choices(document, k=sample_size))
        resampled_labels.append(list(repeat(list(labels[n]), sample_size)))
    resampled_corpuses = [sent for doc in resampled_corpuses for sent in doc]
    resampled_labels = [label for doc in resampled_labels for label in doc]
    return np.array(resampled_corpuses), np.array(resampled_labels)


def size(sequence):
    """produces length vector in shape [batch_size, length] for each padded
    sequence of its true length excluding zero vectors.
    sequence of vectors comes in shape of [batch_size, STEP_SIZE, dimensions]
    """
    flag = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    # [BATCH_SIZE, STEP_SIZE] with 0 or 1
    length = tf.reduce_sum(flag, axis=1)
    # [BATCH_SIZE, 1] with number of 1s used as true lengths
    return tf.cast(length, tf.int32)


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
    label = tf.convert_to_tensor(label_encoded, dtype=tf.int16)
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
          valid_label_batch, optimiser, metric, loss):
    for global_step in range(n):
        train_sent, train_label = \
                    sess.run(fetches=[train_sent_batch, train_label_batch])
        train_accuracy, train_loss = \
            sess.run(fetches=[optimiser, metric, loss],
                     feed_dict={x: train_sent, y_: train_label})
        print("step {0} of {3}, train accuracy: {1:.4f}"
              " log loss: {2:.4f}".format(global_step, train_accuracy,
                                          train_loss, n))

        if global_step and global_step % 10 == 0:
            valid_sent, valid_label = \
                    sess.run(fetches=[valid_sent_batch, valid_label_batch])
            valid_accuracy, valid_loss = \
                sess.run(fetches=[metric, loss],
                         feed_dict={x: valid_sent, y_: valid_label})

            print("\nstep {0} of {3}, valid accuracy: {1:.4f}"
                  " log loss: {2:.4f}\n".format(global_step, valid_accuracy,
                                                valid_loss, n))


STATE_SIZE = 24
STEP_SIZE = 200
N_CLASS = len(labels)

BATCH_SIZE = 50
EPOCH = 200

corpus_encoder = WordEmbedding(raw_corpus=corpus,
                               zero_pad=True,
                               pad_length=STEP_SIZE,
                               n=None)
encoded_corpus = corpus_encoder.encode()

label_encoder = preprocessing.LabelBinarizer().fit(labels)
encoded_labels = label_encoder.transform(labels)

X_train, X_test, y_train, y_test = \
                        model_selection.train_test_split(encoded_corpus,
                                                         encoded_labels,
                                                         test_size=.2)

embedding_matrix = corpus_encoder.vectorize()
embed_shape = embedding_matrix.shape

x = tf.placeholder(dtype=tf.int32, shape=(None, STEP_SIZE), name='feature')
y_ = tf.placeholder(dtype=tf.int16, shape=(None, N_CLASS), name='label')

embeddings = \
        tf.get_variable(name='embeddings',
                        shape=embed_shape,
                        initializer=tf.constant_initializer(embedding_matrix),
                        trainable=False)

rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

W_softmax = tf.get_variable(
                    name='W_softmax',
                    shape=[STATE_SIZE, N_CLASS],
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
b_softmax = tf.get_variable(
                    name='b_softmax',
                    shape=[N_CLASS],
                    initializer=tf.constant_initializer(0.0))

cell = tf.contrib.rnn.BasicLSTMCell(STATE_SIZE)
outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                         inputs=rnn_inputs,
                                         sequence_length=size(rnn_inputs),
                                         dtype=tf.float32)

# outputs in shape [BATCH_SIZE, STEP_SIZE, STATE_SIZE]
outputs = tf.transpose(a=outputs, perm=[1, 0, 2])
# reshaped to [STEP_SIZE, BATCH_SIZE, STATE_SIZE], slice for last step only
logits = tf.matmul(outputs[-1], W_softmax) + b_softmax
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_)

loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

with sess, tf.device('/cpu:0'):
    train_sent_batch, train_label_batch = \
        batch_generator(*enqueue(*resample(X_train, y_train)), BATCH_SIZE)
    valid_sent_batch, valid_label_batch = \
        batch_generator(*enqueue(*resample(X_test, y_test)), BATCH_SIZE)
    train(5000, x, y_, train_sent_batch, train_label_batch, valid_sent_batch,
          valid_label_batch, train_step, accuracy, loss)
