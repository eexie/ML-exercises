import tensorflow as tf
import numpy as np
import input_data

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model (X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden): #input, weights, keep_probs
	X = tf.nn.dropout(X, p_drop_input) #randomly remove pixels from input images

	h = tf.nn.relu(tf.matmul(X, w_h)) #activation function
	h = tf.nn.dropout(h, p_drop_hidden) #regularization

	h2 = tf.nn.relu(tf.matmul(h, w_h2))
	h2 = tf.nn.dropout(h2, p_drop_hidden)

	return tf.matmul(h2, w_o)	# basic mlp, 2 layers, can be seen as 2 stacked logistic regressions

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(100):
	for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
		train_op.run(feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_input:0.8, p_keep_hidden: 0.5})
	print i, np.mean(np.argmax(teY, axis = 1) == predict_op.run(feed_dict={X: teX, y:teY, p_keep_input:1.0, p_keep_hidden:1.0}))
sess.close()