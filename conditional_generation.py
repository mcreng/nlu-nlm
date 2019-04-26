import pickle

import numpy as np
import tensorflow as tf
from gensim import models

import load_embedding as LE
load_embedding = LE.load_embedding


## Constants ##
num_units = 1024
proj_units = 512
batch_size = 64
timesteps = 30
num_vocab = 20000
embedding_dim = 100
###############


with open('word_2_idx.pkl', 'rb') as f:
    word_2_idx = pickle.load(f)


with open('idx_2_word.pkl', 'rb') as f:
    idx_2_word = pickle.load(f)


"""
Load and process continuation data into prefixes and numpy arrays of word indices arrays
"""
def continuation_data():
    with open('sentences.continuation', 'r') as f:
        lines = f.readlines()

        prefixes = [line.strip().split() for line in lines]

        # add <bos> token not explicitly in the file
        lines = ('<bos> ' + line.strip() for line in lines)

        # pad short sentences to 30 tokens
        # will be replaced by tokens generated afterwards
        lines = (line + ' <pad>' * (30 - len(line.split(' '))) for line in lines)

    sentences = []
    for line in lines:
        # <unk> token maps to 0
        ids = list(map(lambda w: word_2_idx.get(w, 0), line.split()))
        sentences.append(ids)

    return prefixes, np.asarray(sentences, dtype=np.int32)


tf.reset_default_graph()

lstm = tf.nn.rnn_cell.LSTMCell(num_units=num_units,
                               initializer=tf.contrib.layers.xavier_initializer())

X = tf.placeholder(tf.int32, [batch_size, timesteps])
embedding = tf.placeholder(tf.float32, [batch_size, timesteps, embedding_dim])

x = tf.unstack(embedding, timesteps, 1)

state = lstm.zero_state(batch_size, 'float')

states = []
for t in range(timesteps-1):
    p, state = lstm(x[t], state)
    states.append(state.h)

states = tf.stack(states)

dim_reduction = tf.layers.dense(
    states, proj_units, kernel_initializer=tf.contrib.layers.xavier_initializer())

logits = tf.layers.dense(
    dim_reduction, num_vocab, kernel_initializer=tf.contrib.layers.xavier_initializer())

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.transpose(X[:, 1:]), logits=logits))

optimizer = tf.train.AdamOptimizer()
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_step = optimizer.apply_gradients(zip(gradients, variables))


with tf.Session() as sess:
    full_embeddings = load_embedding(sess, word_2_idx, "wordembeddings-dim100.word2vec", embedding_dim, num_vocab)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('./savedC'))

    prefixes, x_val = continuation_data()
    num_sentences = x_val.shape[0]

    for i in range(num_sentences):
        print(f'Generating sentence {i}')

        # retrieve i-th sentence from continuation data
        sentence = x_val[i]
        sentence = np.reshape(sentence, (1, timesteps))

        # pad (batch_size - 1) fake sentences for network input format
        x_batch = np.append(sentence,
                            np.zeros(shape=(batch_size - 1, timesteps),
                                     dtype=np.int8),
                            axis=0)

        while len(prefixes[i]) < 20 and prefixes[i][-1] != '<eos>':
            x_batch = x_batch.reshape(batch_size * timesteps)

            pretrained_embeddings = full_embeddings[x_batch]
            pretrained_embeddings = np.reshape(pretrained_embeddings, (batch_size, timesteps, embedding_dim))

            x_batch = np.reshape(x_batch, (batch_size, timesteps))

            output = sess.run([logits],
                              feed_dict={X: x_batch, embedding: pretrained_embeddings})[0]

            # adjust output to batch_size x timesteps x num_vocab
            output = output.transpose(1, 0, 2)

            # take the first sentence
            output = output[0]

            # transform to idx representation
            predicted_sentence = np.argmax(output, axis=1).tolist()

            # retrieve the next predicted token
            next_token_idx = predicted_sentence[len(prefixes[i])]

            # update the network input
            # the 0-th sentence of x_batch corresponds to the actual prefix
            x_batch[0, len(prefixes[i]) + 1] = next_token_idx

            # update the prefix with token generated
            next_token_word = idx_2_word[next_token_idx]
            prefixes[i].append(next_token_word)

    with open('group27.continuation', 'w') as f:
        for prefix in prefixes:
            f.write(' '.join(prefix) + '\n')
