import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 101) #create some random data
trY = 2*trX+np.random.randn(*trX.shape)*0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X,w):
	return tf.mul(X,w)

w = tf.Variable(0.0, name = "weights")
init = tf.initialize_all_variables()
y_model = model(X,w)

cost = (tf.pow(Y-y_model, 2)) #use sqr error for cost function

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()

sess.run(init)

for i in range(100):
    for (x, y) in zip(trX, trY): 
        sess.run(train_op, feed_dict={X: x, Y: y})

print(sess.run(w))
sess.close()