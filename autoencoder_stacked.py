import tensorflow as tf
import numpy as np

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
samples = mnist.train.images
validation_samples = mnist.validation.images



with tf.Graph().as_default():
    input = samples
    lr = 0.01
    in_dim = input.shape[1]
    out_dim = in_dim
    h_dim = 100
    h2_dim = 50
    encoding_matrices=[]
    biases=[]

    targets_train = input

    x = tf.placeholder(dtype='float',shape=[None,in_dim],name='x')

    maxval_ih = tf.sqrt(6.0/(in_dim+h_dim))
    maxval_ho = tf.sqrt(6.0/(out_dim+h_dim))

    w=tf.Variable(tf.random_uniform(shape=[in_dim,h_dim],minval=-maxval_ih,maxval=maxval_ih,dtype='float'),name='w1')
    wt=tf.Variable(tf.random_uniform(shape=[h_dim,in_dim],minval=-maxval_ih,maxval=maxval_ih,dtype='float'),name='w2')
    w2=tf.Variable(tf.random_uniform(shape=[h_dim,h2_dim],minval=-maxval_ih,maxval=maxval_ih,dtype='float'),name='w1')
    w2t=tf.Variable(tf.random_uniform(shape=[h2_dim,h_dim],minval=-maxval_ih,maxval=maxval_ih,dtype='float'),name='w2')
    b=tf.Variable(tf.random_uniform(shape=[h_dim],minval=-0.01,maxval=0.01,dtype='float'),name='b1')
    bt=tf.Variable(tf.random_uniform(shape=[in_dim],minval=-0.01,maxval=0.01,dtype='float'),name='b1')
    b2=tf.Variable(tf.random_uniform(shape=[h2_dim],minval=-0.01,maxval=0.01,dtype='float'),name='b2')
    b2t=tf.Variable(tf.random_uniform(shape=[h_dim],minval=-0.01,maxval=0.01,dtype='float'),name='b1')
    batchSize = 10000
    sess = tf.InteractiveSession()
    # sess.run(tf.initialize_all_variables())

    # w = encoding_matrices[0]
    # b = biases[0]
    l1 = tf.nn.tanh(tf.matmul(x,w) + b)
    l1r = tf.nn.tanh(tf.matmul(l1, wt)+bt)


    #l2 = l2_logit
    targets = tf.placeholder(dtype='float',shape=[None,out_dim],name='targets')
    cost1 = tf.nn.l2_loss(targets - l1r)
    cost1 = tf.reduce_sum(cost1) / 60000 #samples
    

    t_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=cost1, var_list = [w, wt, b, bt])
    
    targets2 = l1
    l2 = tf.nn.tanh(tf.matmul(targets2, w2)+b2)
    l2r = tf.nn.tanh(tf.matmul(l2, w2t)+b2t)
    cost2 = tf.nn.l2_loss(targets2 - l2r)
    cost2 = tf.reduce_mean(cost2)/60000
    t_step2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=cost2, var_list = [w2, w2t, b, bt])


    l1rr = tf.nn.tanh(tf.matmul(l2r, wt)+bt)
    cost3 = tf.nn.l2_loss(targets - l1rr)
    cost3 = tf.reduce_mean(cost3)/60000
    train_op = tf.train.AdamOptimizer(lr).minimize(cost3)


    w_hist = tf.histogram_summary("weights", w)
    b_hist = tf.histogram_summary("biases", b)
    writer = tf.train.SummaryWriter("/tmp/mnist_logs_3", sess.graph_def)
    merged = tf.merge_all_summaries()
    sess.run(tf.initialize_all_variables())
    c0 = 0
    while True:
        for batch in range (0,input.shape[0],batchSize):
            batch_end = min(input.shape[0],batch+batchSize) + 1

            t_step.run(feed_dict={x:input[batch:batch_end],targets:targets_train[batch:batch_end]})
        c1 = cost1.eval(feed_dict={x:input,targets:targets_train})
        if ((c1-c0)>0.1):
            print c1
            c0 = c1
        else:
            break
    # w = encoding_matrices[0]
    # w2 = encoding_matrices[1]
    # sess.run(tf.initialize_all_variables())
    c0 = 0
    while True:
        for batch in range (0,input.shape[0],batchSize):
            batch_end = min(input.shape[0],batch+batchSize) + 1

            t_step2.run(feed_dict={x:input[batch:batch_end]})
        c2 = cost2.eval(feed_dict={x:input,targets:targets_train})
        if ((c2-c0)>0.1):
            print c2
            c0 = c2
        else:
            break


    for i in range(0,10):
        print i
    # sess.run(tf.initialize_all_variables())
        for batch in range (0,input.shape[0],batchSize):
            batch_end = min(input.shape[0],batch+batchSize) + 1
            summary_str = sess.run(merged, feed_dict={x:input[batch:batch_end],targets:targets_train[batch:batch_end]})
            writer.add_summary(summary_str, i)
            train_op.run(feed_dict={x:input[batch:batch_end],targets:targets_train[batch:batch_end]})
            c3 = cost3.eval(feed_dict={x:input[batch:batch_end],targets:targets_train[batch:batch_end]})
        v = cost3.eval(feed_dict={x:validation_samples,targets:validation_samples})
        print c3, v

