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
flags.DEFINE_integer('PRF',0,'calculate PRF')
flags.DEFINE_float('learning_rate',1e-3,'learning rate')
flags.DEFINE_float('dropout',0.5,'dropout')
flags.DEFINE_string('data_path',None,'data path')
flags.DEFINE_string('embedding_path','./embeddings','embedding_path')
flags.DEFINE_string('saver_path','./model_saver','saver_path')
flags.DEFINE_string('output_path','./output.txt','prediction_output_path')
flags.DEFINE_string('label_dict_path','./id_to_label_dict','id_to_label_dict_path')
flags.DEFINE_string('word_dict_path','./id_to_word_dict','id_to_word_dict_path')

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
		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=1)

		# Get lstm cell output
		outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
		outputs = tf.reshape(outputs,[-1,FLAGS.n_hidden])

		logits = tf.matmul(outputs, weights) + biases
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_))
		optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(cost)

	correct_prediction = tf.nn.in_top_k(logits, y_, 1)
	values, indices = tf.nn.top_k(logits, 1)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	correct = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	if FLAGS.PRF == 0:
		with tf.Session(config = config) as sess:
			saver = tf.train.Saver()
			best_acc = 0
			sess.run(tf.initialize_all_variables())
			for step in range(FLAGS.epoch_size * FLAGS.epoch_step):
				batch_x,batch_y = train_data.next_batch()
				sess.run(optimizer, feed_dict = {x_:batch_x,y_:batch_y,output_keep_prob:1-FLAGS.dropout})
				#if step % 1000 == 0:
				#	print cost.eval(feed_dict = {x_:batch_x,y_:batch_y,output_keep_prob:1})
				if step % FLAGS.epoch_size == 0:
					num = 0
					cor_num = 0
					for i in range(0, test_data.sentence_num-1):
						test_x, test_y = test_data.next_batch()
						num += len(test_y)
						_, correct_num = sess.run([optimizer,correct],feed_dict = \
						{x_:test_x, y_:test_y, output_keep_prob:1})
						cor_num += correct_num
					test_accuracy = cor_num/num
					if test_accuracy >= best_acc:
						saver.save(sess,FLAGS.saver_path)
					print "step %d , test_accuracy: %g" % (step,test_accuracy)
	else:
		with tf.Session(config = config) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, FLAGS.saver_path)
			label_dict = pkl.load(open(FLAGS.label_dict_path,'r'))
			word_dict = pkl.load(open(FLAGS.word_dict_path,'r'))
			out = open(FLAGS.output_path,'word_dict_path')
			num = 0
			cor_num = 0
			for i in range(0, test_data.sentence_num-1):
				test_x, test_y = test_data.next_batch()
				num += len(test_y)

				_, correct_num = sess.run([optimizer,correct],feed_dict = \
						{x_:test_x, y_:test_y, output_keep_prob:1})
				cor_num += correct_num

				ind = indices.eval(feed_dict = {x_:test_x, y_:test_y, output_keep_prob:1})
				for j in range(0,len(test_y)-1):
					out.write(label_dict[test_y[j]] + '\t' + label_dict[ind[j][0]] + '\n')
				out.write('\n')
				out.flush()
			print "test_accuracy: %g" % (cor_num/num)
			out.close()


def main(_):
	dynamic_rnn()

if __name__ == '__main__':
	tf.app.run()
