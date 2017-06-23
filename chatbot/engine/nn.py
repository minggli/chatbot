"""
    nn (needs modularisation and further refactoring)

    sequence classification of variable lengths through
    Long-short Term Memory (LSTM) which models surrounding context of words.

    Words are represented in 300-dimension vectors to allow generalization over
    semantic redundancy.

    Distributed Representations of Words and Phrases and their Compositionality
    Mikolov et al. 2013
"""

import numpy as np
import tensorflow as tf

from sklearn import preprocessing

from chatbot.engine import corpus, labels
from chatbot.engine.helper import (flatten_split_resample, batch_generator,
                                   find_last, size, enqueue, train,
                                   save_session, restore_session)
from chatbot.nlp.embedding import WordEmbedding
from chatbot.serializers import feed_conversation
from chatbot.settings import (CacheSettings, FORCE, STATE_SIZE, STEP_SIZE,
                              BATCH_SIZE, MAX_WORDS, MAX_STEPS, VERBOSE)

corpus_encoder = WordEmbedding(top=MAX_WORDS)
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
keep_prob = tf.cond(is_train, lambda: tf.constant(.5), lambda: tf.constant(1.))

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
              embeddings=embeddings,
              limit=5,
              decision_boundary=.65):
    """produce probabilities of most probable topic"""

    encoder, original_pad_length = encoder.fit([question]), encoder.pad_length
    encoded_query = encoder.encode(pad_length=original_pad_length)
    encoded_query = np.array(encoded_query).reshape(-1, original_pad_length)
    embedded_query = encoder.vectorize()

    class_prob = sess.run(fetches=probs,
                          feed_dict={query: encoded_query,
                                     embeddings: embedded_query})

    class_prob = class_prob.mean(axis=0).tolist()
    samples = [(class_, class_prob[k]) for k, class_ in enumerate(classes)]

    return feed_conversation(samples, limit, decision_boundary)
