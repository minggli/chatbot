"""
    lstm (needs modularisation and further refactoring)

    sequence classification of variable lengths through
    Long-short Term Memory (LSTM) which models surrounding context of words.

    Words are represented in 300-dimension vectors to allow generalization over
    semantic redundancy.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""
import os
import random

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from functools import wraps
from sklearn import model_selection, preprocessing

from chatbot.nlp.embedding import WordEmbedding
from chatbot.serializers import feed_conversation

from . import corpus, labels

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
    length = tf.reduce_sum(flag, axis=1)
    return tf.cast(length, tf.int32)


def find_last(outputs, length):
    """zero padded sequences will produce zero vectors after sequence_length
    required using dynamic unrolling dynamic_rnn API. this function locates
    the last non-zero output for calculating logits
    """
    # outputs in shape [BATCH_SIZE, STEP_SIZE, STATE_SIZE], length [BATCH_SIZE]
    b, t, st = tf.unstack(tf.shape(outputs))
    flat_index = tf.range(b) * t + (length - 1)
    zeros = tf.zeros(shape=[b], dtype=tf.int32)
    non_neg_flat_index = tf.maximum(zeros, flat_index)
    flat_outputs = tf.reshape(outputs, [-1, st])
    last_outputs = tf.gather(flat_outputs, non_neg_flat_index)
    return last_outputs


def timeit(func):
    """calculate time for a function to complete"""
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print('function {0} took {1:0.3f} s'.format(
              func.__name__, (end - start) * 1))
        return output
    return wrapper


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
          valid_label_batch, keep_prob, optimiser, metric, loss, verbose):

    for global_step in tqdm(range(n), miniters=1):
        train_sent, train_label = \
                    sess.run(fetches=[train_sent_batch, train_label_batch])
        _, train_accuracy, train_loss = \
            sess.run(fetches=[optimiser, metric, loss],
                     feed_dict={x: train_sent, y_: train_label, keep_prob: .8})

        if verbose:
            print("step {0} of {3}, train accuracy: {1:.4f} log loss: {2:.4f}"
                  .format(global_step, train_accuracy, train_loss, n))

        if global_step and global_step % 10 == 0:
            valid_sent, valid_label = \
                    sess.run(fetches=[valid_sent_batch, valid_label_batch])
            valid_accuracy, valid_loss = sess.run(
                    fetches=[metric, loss],
                    feed_dict={x: valid_sent, y_: valid_label, keep_prob: 1})

            if verbose:
                print("step {0} of {3}, valid accuracy: {1:.4f} "
                      "log loss: {2:.4f}".format(global_step, valid_accuracy,
                                                 valid_loss, n))


N_DOC = 20
corpus = corpus[:N_DOC]
labels = labels[:N_DOC]

STATE_SIZE = 24
STEP_SIZE = 30
N_CLASS = len(labels)
BATCH_SIZE = 50
EPOCH = 200

corpus_encoder = WordEmbedding(top=None).fit(corpus)
encoded_corpus = corpus_encoder.encode(zero_pad=True, pad_length=STEP_SIZE)

l_encoder = preprocessing.LabelBinarizer().fit(labels)
encoded_labels, classes = l_encoder.transform(labels), l_encoder.classes_

embedding_matrix = corpus_encoder.vectorize()
embed_shape = embedding_matrix.shape

x = tf.placeholder(dtype=tf.int32, shape=(None, STEP_SIZE), name='feature')
y_ = tf.placeholder(dtype=tf.uint8, shape=(None, N_CLASS), name='label')
v = tf.placeholder(dtype=tf.float32, shape=(None, 300), name='vectors')
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_rate')

embeddings = tf.get_variable(name='embedd', shape=embed_shape, trainable=False)

rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

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

last = find_last(outputs, sent_length)
logits = tf.matmul(last, W_softmax) + b_softmax
# logits in shape [BATCH_SIZE, N_CLASS]
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_)
# softmax cross entropy in shape [BATCH_SIZE, ]
# TODO mask padded loss to accelerate training
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

probs = tf.nn.softmax(logits)
correct = tf.equal(tf.argmax(probs, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
init_embedd = tf.assign(embeddings, v, validate_shape=False)

sess = tf.Session()
sess.run(init)
sess.run(init_embedd, feed_dict={v: embedding_matrix})

with sess.as_default(), tf.device('/cpu:0'):
    data = flatten_split_resample(encoded_corpus, encoded_labels,
                                  valid_ratio=.2, sample_size=1000)
    train_sent_batch, train_label_batch = \
        batch_generator(*enqueue(*data[0]), BATCH_SIZE)
    valid_sent_batch, valid_label_batch = \
        batch_generator(*enqueue(*data[1]), BATCH_SIZE)
    train(300, x, y_, train_sent_batch, train_label_batch, valid_sent_batch,
          valid_label_batch, keep_prob, train_step, accuracy, loss, False)


def inference(query, sess, encoder=corpus_encoder, classes=classes, limit=5,
              decision_boundary=.85):
    """produce probabilities of most probable topic"""
    encoder, original_pad_length = encoder.fit([query]), encoder.pad_length
    encoded_query = encoder.encode(pad_length=original_pad_length)
    encoded_query = np.array(encoded_query).reshape(-1, original_pad_length)
    sess.run(fetches=init_embedd, feed_dict={v: encoder.vectorize()})
    probabilities = sess.run(fetches=probs,
                             feed_dict={x: encoded_query, keep_prob: 1})
    class_prob = np.mean(probabilities, axis=0).tolist()
    samples = [(class_, class_prob[k]) for k, class_ in enumerate(classes)]
    return feed_conversation(samples, limit, decision_boundary)
