import tensorflow as tf
import numpy as np

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
samples = mnist.train.images
validation_samples = mnist.validation.images
layer_sizes=[300,150,75,35,18,9,5,2]
input = samples
targets_train = input
lr = 0.01
in_dim = input.shape[1]
out_dim = in_dim
e_w=[] #encoding weights
b_in=[]
b_out =[]
layers = []
train_ops = []
x = tf.placeholder(dtype='float',shape=[None,in_dim],name='x')
targets = tf.placeholder(dtype='float',shape=[None,out_dim],name='targets')
cur_input = x
batchSize = 10000
sess = tf.InteractiveSession()

for dim in layer_sizes:    

    maxval_ih = tf.sqrt(6.0/(in_dim+dim))
    maxval_ho = tf.sqrt(6.0/(out_dim+dim))

    e_w.append(tf.Variable(tf.random_uniform(shape=[in_dim,dim],minval=-maxval_ih,maxval=maxval_ih,dtype='float'),name='w1'))
    b_in.append(tf.random_uniform(shape=[dim],minval=-0.01,maxval=0.01,dtype='float'),name='b1'))
    b_out.append(tf.Variable(tf.random_uniform(shape=[in_dim],minval=-0.01,maxval=0.01,dtype='float'),name='b1'))

    in_dim = dim
    wt = tf.transpose(e_w[dim])
    layers.append(tf.nn.tanh(tf.matmul(cur_input, e_w[dim]) + b_in[dim]))
    lr = tf.nn.tanh(tf.matmul(layers[dim], tf.transpose(e_w[dim])+b_out[dim]))

    #l2 = l2_logit
    if (dim==1):
        cost = tf.nn.l2_loss(targets - lr)
    else:
        cost = tf.nn.l2_loss(layers[dim-1] - lr)
    cost = tf.reduce_sum(cost) / 60000 #samples
    
    tran_ops.append(tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=cost, var_list = [e_w[dim], wt, b_in[dim], b_out[dim]]))
    

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

