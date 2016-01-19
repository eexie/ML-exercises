import tensorflow as tf
import numpy as np
import input_data

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model (X, w_h, w_0):
	h = tf.nn.sigmoid(tf.matmul(X, w_h)) 
	return tf.matmul(h, w_0)	# basic mlp, 2 layers, can be seen as 2 stacked logistic regressions


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])
w_0 = init_weights([625, 10])

py_x = model(X, w_h, w_0)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(py_x,1)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100):
	for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
		sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
	print i, np.mean(np.argmax(teY, axis = 1) == 
				sess.run(predict_op, feed_dict={X:teX, Y:teY}))


sess.close();
