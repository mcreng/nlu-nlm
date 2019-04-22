import math
import pickle

import numpy as np
import tensorflow as tf


## Constants ##
epochs = 1
num_units = 512
batch_size = 64
timesteps = 30
num_vocab = 20000
embedding_dim = 100
###############


with open('word_2_idx.pkl', 'rb') as f:
    word_2_idx = pickle.load(f)


"""
Load preprocessed training data into numpy arrays of word indices arrays
"""
def training_data():
    with open('sentences.train.proc', 'r') as f:
        sentences = []
        for sentence in f.readlines():
            ids = list(map(lambda w: word_2_idx[w], sentence.split()))
            sentences.append(ids)
        return np.asarray(sentences, dtype=np.int32)


"""
Randomly sample batch_size data points from the data
"""
def mini_batch(data, batch_size):
    return data[np.random.choice(data.shape[0], batch_size, replace=False), :]


"""
Load and process testing data into numpy arrays of word indices arrays
"""
def testing_data():
    with open('sentences_test.txt', 'r') as f:
        lines = f.readlines()

        # add <bos> and <eos> tags
        lines = ('<bos> ' + line.strip() + ' <eos>' for line in lines)
        # remove lines with length greater than 30
        lines = (line for line in lines if len(line.split(' ')) <= 30)
        # pad short sentences to 30 tokens
        lines = (line + ' <pad>' * (30 - len(line.split(' '))) for line in lines)

    sentences = []
    for line in lines:
        # <unk> token maps to 0
        ids = list(map(lambda w: word_2_idx.get(w, 0), line.split()))
        sentences.append(ids)

    return np.asarray(sentences, dtype=np.int32)


tf.reset_default_graph()

lstm = tf.nn.rnn_cell.LSTMCell(num_units=num_units,
                               initializer=tf.contrib.layers.xavier_initializer())

X = tf.placeholder(tf.int32, [batch_size, timesteps])

one_hot = tf.one_hot(X, num_vocab)
embedding = tf.layers.dense(one_hot,
                            embedding_dim,
                            use_bias=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

x = tf.unstack(embedding, timesteps, 1)

state = lstm.zero_state(batch_size, 'float')

states = []
for t in range(timesteps-1):
    p, state = lstm(x[t], state)
    states.append(state[1])

states = tf.stack(states)

logits = tf.layers.dense(
    states, num_vocab, kernel_initializer=tf.contrib.layers.xavier_initializer())

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.transpose(X[:, 1:]), logits=logits))

optimizer = tf.train.AdamOptimizer()
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_step = optimizer.apply_gradients(zip(gradients, variables))


##### Compute perplexity ######

z = tf.unstack(tf.one_hot(X, num_vocab), timesteps, 1)

pad_id = word_2_idx['<pad>']
loge2 = float(math.log(2))

# adjust the pad token to zero that it should not be computed in the final result
z = (tf.stack(z))[1:, :, :]
adjuster = np.ones((num_vocab),dtype=np.float32)

adjuster[pad_id] = 0
adjuster = tf.convert_to_tensor(adjuster)

probs = tf.reduce_sum(tf.multiply(tf.nn.softmax(logits), z), axis=2)
log_probs = tf.div(tf.log(probs), loge2)

z = tf.multiply(z, adjuster)
z = tf.reduce_sum(z, axis=2)
log_probs = tf.multiply(log_probs, z)

n_stack = tf.reduce_sum(z, axis=0)
perplexity = tf.pow(float(2), tf.multiply(tf.div(tf.reduce_sum(log_probs, axis=0), n_stack), float(-1)))

###### End of computing perplexity


with tf.Session() as sess:
    # training part
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    x_val = training_data()

    for epoch in range(epochs):
        iterations = x_val.shape[0] // batch_size
        for i in range(iterations):
            x_batch = mini_batch(x_val, batch_size)
            val_loss, _ = sess.run([loss, train_step], feed_dict={X: x_batch})
            print(f'Epoch {epoch}, Iteration {i}/{iterations}, Loss: {val_loss}')

        saver.save(sess, './savedA/model_A.ckpt')


    # testing part
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('./savedA'))

    x_val = testing_data()
    num_sentences = x_val.shape[0]

    # pad testing data to have length of multiple of batch_size
    x_augmented = np.append(x_val,
                            np.zeros(shape=(batch_size - num_sentences % batch_size, timesteps)),
                            axis=0)

    results = []
    for i in range(0, num_sentences, batch_size):
        x_batch = x_augmented[i : i+batch_size]
        perplexities = sess.run([perplexity], feed_dict={X: x_batch})
        results += perplexities[0].tolist()

    # remove the augmented part
    results = results[:num_sentences]

    with open('group27.perplexityA', 'w') as f:
        for result in results:
            f.write(f'{result}\n')
