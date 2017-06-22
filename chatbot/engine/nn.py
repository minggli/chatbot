"""
    nn (needs modularisation and further refactoring)

    sequence classification of variable lengths through
    Long-short Term Memory (LSTM) which models surrounding context of words.

    Words are represented in 300-dimension vectors to allow generalization over
    semantic redundancy.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""
import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from datetime import datetime
from sklearn import model_selection, preprocessing

from chatbot.engine import corpus, labels
from chatbot.nlp.embedding import WordEmbedding
from chatbot.nlp.sparse import NLPPipeline
from chatbot.serializers import feed_conversation
from chatbot.settings import (CacheSettings, NLP_ATTRS, FORCE, STATE_SIZE,
                              STEP_SIZE, BATCH_SIZE, MAX_WORDS, MAX_STEPS,
                              VERBOSE)


def resample(docs, labels, sample_size):
    unique_labels = np.unique(labels)
    indice_array = list()
    for label in unique_labels:
        indice_label = np.where(labels == label)[0].tolist()
        indice_array.extend(
            np.random.choice(indice_label, size=sample_size).tolist())
    return (docs[indice_array], labels[indice_array])


def flatten_split_resample(encoded_corpuses, encoded_labels,
                           valid_ratio=.2,
                           sample_size=5000):
    """break documents into sentences and augment, and one-hot encode labels"""

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

    return resample(X_train, y_train, sample_size),\
        resample(X_test, y_test, sample_size)


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


def enqueue(sent_encoded, label_encoded, num_epochs=None, shuffle=True):
    """returns an Ops Tensor with queued sentence and label pair"""
    sent = tf.convert_to_tensor(sent_encoded, dtype=tf.int32)
    label = tf.convert_to_tensor(label_encoded, dtype=tf.uint8)
    input_queue = tf.train.slice_input_producer(
                                tensor_list=[sent, label],
                                num_epochs=num_epochs,
                                shuffle=shuffle)
    return input_queue


def batch_generator(sent_queue, label_queue, batch_size=None, threads=8):
    return tf.train.shuffle_batch(
                    tensors=[sent_queue, label_queue],
                    batch_size=batch_size,
                    num_threads=threads,
                    capacity=batch_size * 10,
                    min_after_dequeue=batch_size,
                    allow_smaller_final_batch=True)


def restore_session(sess, path):
    """restore hard trained model for predicting."""
    eval_saver = \
        tf.train.import_meta_graph(tf.train.latest_checkpoint(path) + '.meta')
    eval_saver.restore(sess, tf.train.latest_checkpoint(path))
    print('{} restored successfully.'.format(tf.train.latest_checkpoint(path)))


def save_session(sess, path, sav):
    """save hard trained model for future predicting."""
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = sav.save(sess, path + "model_{0}.ckpt".format(now))
    print("Model saved in: {0}".format(save_path))


def train(n, sess, is_train, optimiser, metric, loss, verbose):

    for global_step in tqdm(range(n), unit='step', disable=verbose):
        _, train_accuracy, train_loss = \
            sess.run([optimiser, metric, loss], feed_dict={is_train: True})

        if verbose:
            print("step {0} of {3}, train accuracy: {1:.4f} log loss: {2:.4f}"
                  .format(global_step, train_accuracy, train_loss, n))

        if global_step and global_step % 100 == 0:
            valid_accuracy, valid_loss = sess.run(fetches=[metric, loss])
            print("step {0} of {3}, train accuracy: {1:.4f} log loss: {2:.4f}"
                  .format(global_step, train_accuracy, train_loss, n))
            print("step {0} of {3}, valid accuracy: {1:.4f} "
                  "log loss: {2:.4f}".format(global_step, valid_accuracy,
                                             valid_loss, n))

    print("step {0} of {0}, train accuray: {1:.4f}, valid accuracy {2:.4f} "
          "log loss: {3:.4f}".format(n, train_accuracy, valid_accuracy,
                                     train_loss))


nlp_transform = NLPPipeline(attrs=NLP_ATTRS)
corpus = nlp_transform.process(corpus)

corpus_encoder = WordEmbedding(top=MAX_WORDS, language=nlp_transform._nlp)
corpus_encoder.fit(corpus)
encoded_corpus = corpus_encoder.encode(zero_pad=True, pad_length=STEP_SIZE)

l_encoder = preprocessing.LabelBinarizer().fit(labels)
encoded_labels, classes = l_encoder.transform(labels), l_encoder.classes_

embedding_matrix = corpus_encoder.vectorize()
embed_shape = embedding_matrix.shape

data = flatten_split_resample(encoded_corpus, encoded_labels)
train_sent, train_label = batch_generator(*enqueue(*data[0]), BATCH_SIZE)
valid_sent, valid_label = batch_generator(*enqueue(*data[1]), BATCH_SIZE)

query = tf.placeholder_with_default(input=valid_sent,
                                    shape=[None, STEP_SIZE],
                                    name='query')
embeddings = tf.placeholder_with_default(input=embedding_matrix,
                                         shape=[None, embed_shape[-1]],
                                         name='embeddings')
is_train = tf.placeholder_with_default(input=False,
                                       shape=[],
                                       name='is_train_or_valid')

feature_feed = tf.cond(is_train, lambda: train_sent, lambda: query)
label_feed = tf.cond(is_train, lambda: train_label, lambda: valid_label)
keep_prob = tf.cond(is_train, lambda: tf.constant(.8), lambda: tf.constant(1.))

W_softmax = tf.get_variable(name='W',
                            shape=[STATE_SIZE, len(labels)],
                            initializer=tf.truncated_normal_initializer(
                                        stddev=0.1))
b_softmax = tf.get_variable(name='b',
                            shape=[len(labels)],
                            initializer=tf.constant_initializer(0.0))

word_vectors = tf.nn.embedding_lookup(embeddings, feature_feed)

with tf.device('/gpu:0'):

    cell = tf.nn.rnn_cell.BasicLSTMCell(STATE_SIZE)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,
                                         input_keep_prob=keep_prob,
                                         state_keep_prob=keep_prob,
                                         output_keep_prob=keep_prob)
    sent_length = size(word_vectors)
    outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                             inputs=word_vectors,
                                             sequence_length=sent_length,
                                             dtype=tf.float32)
    last = find_last(outputs, sent_length)
    logits = tf.matmul(last, W_softmax) + b_softmax

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=label_feed)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

    probs = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(probs, 1), tf.argmax(label_feed, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver(max_to_keep=5, var_list=tf.global_variables())

init = tf.global_variables_initializer()
graph_config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=graph_config)
sess.run(init)

with sess.as_default(), tf.device('/cpu:0'):
    try:
        if FORCE:
            raise TypeError
        restore_session(sess, path=CacheSettings.path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

    except TypeError:
        sess.graph.finalize()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        train(MAX_STEPS, sess, is_train, train_step, accuracy, loss, VERBOSE)
        save_session(sess, path=CacheSettings.path, sav=saver)


def inference(question,
              sess=sess,
              encoder=corpus_encoder,
              classes=classes,
              query=query,
              nlp=nlp_transform,
              embeddings=embeddings,
              limit=5,
              decision_boundary=.85):
    """produce probabilities of most probable topic"""
    question = nlp.process([question], prod=True)
    encoder, original_pad_length = encoder.fit(question), encoder.pad_length
    encoded_query = encoder.encode(pad_length=original_pad_length)
    encoded_query = np.array(encoded_query).reshape(-1, original_pad_length)
    embedded_query = encoder.vectorize()

    class_prob = sess.run(fetches=probs,
                          feed_dict={query: encoded_query,
                                     embeddings: embedded_query})

    class_prob = class_prob.mean(axis=0).tolist()

    samples = [(class_, class_prob[k]) for k, class_ in enumerate(classes)]
    samples.sort(key=lambda x: x[1], reverse=True)
    print(samples[:10])
    return feed_conversation(samples, limit, decision_boundary)
