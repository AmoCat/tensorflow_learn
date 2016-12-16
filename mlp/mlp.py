#!usr/env/bin python
#coding:utf-8
import tensorflow as tf
import cPickle as pkl
from dataset import Dataset
import numpy
import csv

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('train', 1, 'train')
flags.DEFINE_string('test_name', 'TestSamples1.csv', 'test_name')
flags.DEFINE_integer('debug', 0, 'debug')

batch_size = 64
learning_rate = 0.001

n_input = 119
n_output = 10
n_hidden_1 = 256
n_hidden_2 = 64

WRITE_MODEL_PATH = "./98.05_model"
READ_MODEL_PATH = "./1.00_model"
    
weights = {
    #"h1" : tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
    #"h2" : tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    #"out" : tf.Variable(tf.truncated_normal([n_hidden_2, n_output]))
    "h1" : tf.get_variable("h1", [n_input, n_hidden_1]),
    "h2" : tf.get_variable("h2", [n_hidden_1, n_hidden_2]),
    "out" : tf.get_variable("out", [n_hidden_2, n_output])
}

biases = {
    #"b1" : tf.Variable(tf.truncated_normal([n_hidden_1])),
    #"b2" : tf.Variable(tf.truncated_normal([n_hidden_2])),
    #"out" : tf.Variable(tf.truncated_normal([n_output]))
    "b1" : tf.get_variable("b1", [n_hidden_1]),
    "b2" : tf.get_variable("b2", [n_hidden_2]),
    "b3" : tf.get_variable("b3", [n_output])

}

def multilayer_perception():
    x = tf.placeholder("float32", [None, n_input])
    y_ = tf.placeholder("int32", [None])

    logits = mlp_core(x)

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_))
    #optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    correct_prediction = tf.nn.in_top_k(logits,y_,1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    values,indices = tf.nn.top_k(logits,1)

    init = tf.initialize_all_variables()

    if FLAGS.train == 1:
        with tf.Session() as sess:
            data = Dataset(1)
    	    sess.run(init)
            best_acc = 0
            saver = tf.train.Saver()
    	    for epoch in range(500):
                for i in range (20000/batch_size + 1):
            	   batch_x,batch_y = data.next_batch(batch_size)
            	   sess.run(optimizer, feed_dict = {x:batch_x, y_:batch_y})

                test_accuracy = accuracy.eval(feed_dict = {x: data.te_np, y_: data.te_label_np})
                print "epoch %d, test_accuracy %g" % (epoch, test_accuracy)
                if test_accuracy > best_acc:
                    saver.save(sess, WRITE_MODEL_PATH)
                    best_acc = test_accuracy
    else:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess,READ_MODEL_PATH)
            if FLAGS.debug == 1:
                data = Dataset(1)
                test_accuracy = accuracy.eval(feed_dict = {x: data.te_np, y_: data.te_label_np})
                print "test_accuracy %g" % (test_accuracy)
                index = indices.eval(feed_dict = {x: data.te_np, y_: data.te_label_np})
            else:
                data = Dataset(0, test_name = FLAGS.test_name)
                label = numpy.zeros([len(data.te_np)])
                index = indices.eval(feed_dict = {x: data.te_np, y_: label})

            with open('Result.csv','w') as csv_file:
                writer = csv.writer(csv_file)
                for row in index:
                    writer.writerow([row[0]])



def mlp_core(x):
	h_1 = tf.nn.relu(tf.matmul(x, weights["h1"]) + biases["b1"])
	h_2 = tf.nn.relu(tf.matmul(h_1, weights["h2"]) + biases["b2"])

	logits = tf.matmul(h_2, weights["out"]) + biases["b3"]
	return logits

if __name__ == "__main__":
    multilayer_perception()