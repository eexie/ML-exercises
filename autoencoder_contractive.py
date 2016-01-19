import tensorflow as tf
import numpy as np

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
samples = mnist.train.images



def DAE(lr):
    input = samples

    in_dim = input.shape[1]
    out_dim = in_dim
    h_dim = 700
    h2_dim = 500
    encoding_matrices=[]
    biases=[]

    targets_train = input

    x = tf.placeholder(dtype='float',shape=[None,in_dim],name='x')

    maxval_ih = tf.sqrt(6.0/(in_dim+h_dim))
    maxval_ho = tf.sqrt(6.0/(out_dim+h_dim))

    encoding_matrices.append(tf.Variable(tf.random_uniform(shape=[in_dim,h_dim],minval=-maxval_ih,maxval=maxval_ih,dtype='float'),name='w1'))
    encoding_matrices.append(tf.Variable(tf.random_uniform(shape=[h_dim,h2_dim],minval=-maxval_ih,maxval=maxval_ih,dtype='float'),name='w2'))
    biases.append(tf.Variable(tf.random_uniform(shape=[h_dim],minval=-0.01,maxval=0.01,dtype='float'),name='b1'))
    biases.append(tf.Variable(tf.random_uniform(shape=[h2_dim],minval=-0.01,maxval=0.01,dtype='float'),name='b2'))
    batchSize = 10000
    sess = tf.InteractiveSession()
    # sess.run(tf.initialize_all_variables())

    w = encoding_matrices[0]
    b = biases[0]
    l1 = tf.nn.tanh(tf.matmul(x,w) + b)
    l2 = tf.nn.sigmoid(tf.matmul(l1-b,tf.transpose(w)))
    #l2 = l2_logit
    targets = tf.placeholder(dtype='float',shape=[None,out_dim],name='targets')
    cost1 = tf.nn.l2_loss(targets - l2)
    cost1 = tf.reduce_sum(cost1) / 60000 #samples
    

    t_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=cost1)
    


    sess.run(tf.initialize_all_variables())
    for i in range(0,10):
        print i
        for batch in range (0,input.shape[0],batchSize):
            batch_end = min(input.shape[0],batch+batchSize) + 1

            t_step.run(feed_dict={x:input[batch:batch_end],targets:targets_train[batch:batch_end]})
        c1 = cost1.eval(feed_dict={x:input,targets:targets_train})
        print c1
    w = encoding_matrices[0]
    w2 = encoding_matrices[1]
    b = biases[0]
    b2 = biases[1]
    targets2 = tf.nn.tanh(tf.matmul(x,w) + b)
    l3 = tf.nn.tanh(tf.matmul(targets2, w2)+b2)
    l4 = tf.nn.sigmoid(tf.matmul(l3-b2, tf.transpose(w2)))
    cost2 = tf.nn.l2_loss(targets2 - l4)
    cost2 = tf.reduce_mean(cost2)/60000
    t_step2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=cost2)
    sess.run(tf.initialize_all_variables())
    for i in range(0,10):
        print i
        for batch in range (0,input.shape[0],batchSize):
            batch_end = min(input.shape[0],batch+batchSize) + 1

            t_step2.run(feed_dict={x:input[batch:batch_end]})
        c2 = cost2.eval(feed_dict={x:input,targets:targets_train})
        print c2
    targets = tf.placeholder(dtype='float',shape=[None,out_dim],name='targets')
    w = encoding_matrices[0]
    w2 = encoding_matrices[1]
    b = biases[0]
    b2 = biases[1]
    l2 = tf.nn.tanh(tf.matmul(x,w) + b)
    l3 = tf.nn.tanh(tf.matmul(l2, w2)+b2)
    l5 = tf.nn.tanh(tf.matmul(l3-b2, tf.transpose(w2)))
    l6 = tf.nn.tanh(tf.matmul(l5-b, tf.transpose(w)))
    cost3 = tf.nn.l2_loss(targets - l6)
    cost3 = tf.reduce_mean(cost3)/60000
    train_op = tf.train.AdamOptimizer(lr).minimize(cost3)
    
    # for i in range(0,10):
    #     print i
    sess.run(tf.initialize_all_variables())
    for batch in range (0,input.shape[0],batchSize):
        batch_end = min(input.shape[0],batch+batchSize) + 1
        # train_op.run(feed_dict={x:input[batch:batch_end],targets:targets_train[batch:batch_end]})
        c3 = cost3.eval(feed_dict={x:input[batch:batch_end],targets:targets_train[batch:batch_end]})
    print c3
    return c3




lr = 0.001
print lr
print DAE(lr)