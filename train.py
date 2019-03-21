import numpy as np
import tensorflow as tf

## Constants ##
num_units = 3
batch_size = 10
timesteps = 7
num_vocab = 20
embed_dim = 10
###############

tf.reset_default_graph()

lstm = tf.nn.rnn_cell.LSTMCell(num_units=num_units,
                               initializer=tf.contrib.layers.xavier_initializer())

X = tf.placeholder('int32', [batch_size, timesteps])
x = tf.unstack(tf.one_hot(X, num_vocab), timesteps, 1)

state = lstm.zero_state(batch_size, 'float')

states = []
for t in range(timesteps-1):
    p, state = lstm(x[t], state)
    states.append(state[1])

states = tf.stack(states)

logits = tf.layers.dense(
    states, num_vocab, kernel_initializer=tf.contrib.layers.xavier_initializer())

loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.transpose(X[:, 1:]), logits=logits))

optimizer = tf.train.AdamOptimizer()  # select optimizer and set learning rate
train_step = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


x_val = np.reshape(np.random.randint(num_vocab, size=batch_size *
                                     timesteps), [batch_size, timesteps])


for _ in range(1000):
    val_loss, _ = sess.run([loss, train_step], feed_dict={X: x_val})
    print(val_loss)
