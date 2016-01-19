import tensorflow as tf
import numpy as np

a = tf.placeholder("float")
b = tf.placeholder("float")
# sess = tf.InteractiveSession()
# y = a*b
# print "%f should equal 2.0" % y.eval(feed_dict={a:1, b:2})
# print "%f should equal 9.0" % y.eval(feed_dict={a:3, b:3})
sess = tf.Session()
y = a*b
print "%f should equal 2.0" % sess.run(y, feed_dict={a:1, b:2})
print "%f should equal 9.0" % sess.run(y,feed_dict={a:3, b:3})
sess.close()