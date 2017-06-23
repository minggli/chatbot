"""
    helper

    helper functions that control the running of classifers
"""
import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from datetime import datetime
from sklearn import model_selection


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
            print("step {0} of {4}, train accuracy: {1:.4f}, "
                  "valid accuracy: {2:.4f} log loss: {3:.4f}".format(
                                            global_step, train_accuracy,
                                            valid_accuracy, valid_loss, n))

    print("step {0} of {0}, train accuray: {1:.4f}, valid accuracy {2:.4f} "
          "log loss: {3:.4f}".format(n, train_accuracy, valid_accuracy,
                                     train_loss))
