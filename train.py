import numpy as np
import tensorflow as tf
import math
import pickle

## Constants ##
num_units = 3
batch_size = 10 # 64
timesteps = 7 # 30
num_vocab = 20 # 20000
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

##### compute perplexity ######
# x: 		(time-1) x batch_size x num_vocab
# logits: 	(time-1) x batch_size x num_vocab

# with open('word_2_idx.pkl','rb') as f:
# 	d = pickle.load(f)
# 	pad_id = d['<pad>']

pad_id = 1 # not

loge2 = float(math.log(2))

# adjust the pad token to zero that it should not be computed in the final result

x = (tf.stack(x))[1:, :, :]
adjuster = np.ones((num_vocab),dtype = np.float32)

adjuster[pad_id] = 0
adjuster = tf.convert_to_tensor(adjuster)

x = tf.multiply(x,adjuster)

probs = tf.reduce_sum(tf.multiply(tf.nn.softmax(logits), x), axis = 2)
log_probs = tf.div(tf.log(probs),loge2)
n_stack = tf.reduce_sum(x,axis=[0,2])
perplexity = tf.pow(float(2), tf.multiply(tf.div(tf.reduce_sum(log_probs, axis=0), n_stack),float(-1)))

###### End of computing perplexity


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


x_val = np.reshape(np.random.randint(num_vocab, size=batch_size *
                                     timesteps), [batch_size, timesteps])


for _ in range(1000):
    val_loss, perplexity, _ = sess.run([loss, perplexity, train_step], feed_dict={X: x_val})
#     val_loss, _ = sess.run([loss, train_step], feed_dict={X: x_val})
    print(val_loss)
    
