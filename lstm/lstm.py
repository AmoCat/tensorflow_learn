#!usr/bin/env python
#coding:utf-8

import os
import sys
import tensorflow as tf 
import numpy as np 
import cPickle as pkl
from read_dataset import Dataset

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size',1,'batch_size')
flags.DEFINE_integer('n_hidden',300,'hidden units')
flags.DEFINE_integer('epoch_step',100,'nums of epochs')
flags.DEFINE_integer('epoch_size',1000,'batchs of each epoch')
flags.DEFINE_integer('n_classes',46,'nums of classes')
flags.DEFINE_integer('emb_size',345823,'embedding size')
flags.DEFINE_integer('word_dim',100,'word dim')
flags.DEFINE_float('learning_rate',1e-3,'learning rate')
flags.DEFINE_float('dropout',0.5,'dropout')
flags.DEFINE_string('data_path',None,'data path')
flags.DEFINE_string('embedding_path','./embeddings','embedding_path')

def dynamic_rnn():
	train_data = Dataset(data_type = 'train')
	test_data = Dataset(data_type = 'test')

	x_ = tf.placeholder(tf.int32,[FLAGS.batch_size,None]) #[FLAGS.batch_size,None]
	y_ = tf.placeholder(tf.int32,[None])
	output_keep_prob = tf.placeholder(tf.float32)
	#x:[batch_size,n_steps,n_input]
	with tf.device('/cpu:0'):
		embedding = pkl.load(open(FLAGS.embedding_path, 'r'))
		x = tf.nn.embedding_lookup(embedding,x_)
	with tf.device('/gpu:1'):	
		weights = tf.get_variable("weights",[FLAGS.n_hidden,FLAGS.n_classes],tf.float32)
		biases = tf.get_variable("biases",[FLAGS.n_classes],tf.float32)

		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden,state_is_tuple=True,activation=tf.nn.relu)
		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=1-FLAGS.dropout)

		# Get lstm cell output
		outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
		outputs = tf.reshape(outputs,[-1,FLAGS.n_hidden])

		logits = tf.matmul(outputs, weights) + biases
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_))
		optimizer = optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(cost)

	correct_prediction = tf.nn.in_top_k(logits,y_,1)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	correct = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config = config) as sess:
		sess.run(tf.initialize_all_variables())
		#sess.run(optimizer,feed_dict = {x_:np.array([3,3]).reshape([1,-1]),y_:np.array([1,1]).reshape([-1]),output_keep_prob:0.5})
		for step in range(FLAGS.epoch_size * FLAGS.epoch_step):
			batch_x,batch_y = train_data.next_batch()
			sess.run(optimizer, feed_dict = {x_:batch_x,y_:batch_y,output_keep_prob:1-FLAGS.dropout})
			#if step % 1000 == 0:
			#	print cost.eval(feed_dict = {x_:batch_x,y_:batch_y,output_keep_prob:1})
			if step % FLAGS.epoch_size == 0:
				num = 0
				cor_num = 0
				for i in range(0,test_data.sentence_num-1):
					test_x, test_y = test_data.next_batch()
					num += len(test_y)
					cor_num += sess.run(correct,feed_dict = \
					{x_:test_x, y_:test_y, output_keep_prob:1-FLAGS.dropout})
				print "step %d , test_accuracy: %g" % (step,cor_num/num)

def main(_):
	dynamic_rnn()

if __name__ == '__main__':
	tf.app.run()
