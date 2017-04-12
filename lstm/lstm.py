#!usr/bin/env python
#coding:utf-8

import os
import sys
import logging
import tensorflow as tf 
import numpy as np 
import cPickle as pkl
from read_dataset import Dataset
from logging_config import logConfig
from util import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('CRF',1,'CRF layer')
flags.DEFINE_integer('batch_size',1,'batch_size')
flags.DEFINE_integer('n_hidden',64,'hidden units')
flags.DEFINE_integer('epoch_step',40,'nums of epochs')
flags.DEFINE_integer('epoch_size',6000,'batchs of each epoch')
flags.DEFINE_integer('n_classes',18,'nums of classes')
flags.DEFINE_integer('emb_size',345823,'embedding size')
flags.DEFINE_integer('word_dim',300,'word dim')
flags.DEFINE_integer('PRF',0,'calculate PRF')
flags.DEFINE_integer('L2',1,'add L2 regularizer')
flags.DEFINE_integer('feature',0,'add pos and ner feature')
flags.DEFINE_integer('BiLSTM',1,'is bi-directional LSTM or not')
flags.DEFINE_integer('ran_emb',0,'add random variable embedding in training process')
flags.DEFINE_integer('pos_emb_size',25,'pos_embedding_size')
flags.DEFINE_integer('ner_emb_size',25,'ner_embedding_size')
flags.DEFINE_integer('feature_emb_size',25,'pos and ner embedding size')
flags.DEFINE_float('learning_rate',1e-3,'learning rate')
flags.DEFINE_float('dropout',0,'dropout')
flags.DEFINE_string('data_path',None,'data path')
flags.DEFINE_string('embedding_path','./minimized_embeddings','embedding_path')
flags.DEFINE_string('saver_path','./model/model-','saver_path')
flags.DEFINE_string('output_path','./output/','prediction_output_path')
flags.DEFINE_string('label_dict_path','./dict/id_to_label_dict','id_to_label_dict_path')
flags.DEFINE_string('word_dict_path','./dict/minimized_id_to_word_dict','id_to_word_dict_path')
flags.DEFINE_string('log_path','./log/','log_path')

POS_NUM = 23
NER_NUM = 9
#WORD_NUM = 345823
WORD_NUM = 1789
BETA_REGUL = 0.01

def crf_evaluate(seq_len,trans_matrix,unary_score,y_pad):
	correct_num = 0
	label_num = 0
	y = np.reshape(y_pad,(FLAGS.batch_size,-1))
	ind = []
	for i in range(0,FLAGS.batch_size):
		viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_score[i], trans_matrix)
		viterbi_sequence_ = viterbi_sequence[:seq_len[i]]
		y_ = y[i][:seq_len[i]]
		correct_num += np.sum(np.equal(viterbi_sequence_,y_))
		label_num += len(y_)
		ind.extend([t] for t in viterbi_sequence_)
		#print type(viterbi_sequence),"viterbi_seq:",viterbi_sequence_
		#print "\n","y:",y_
		#print "correct_num:",correct_num
	return correct_num,label_num,ind

def get_name_tail():
	file_tail = ""
	file_tail += "CRF" if FLAGS.CRF == 1 else ""
	file_tail += "-L2" if FLAGS.L2 == 1 else ""
	file_tail += "-BILSTM" + str(FLAGS.BiLSTM) + "-h" + str(FLAGS.n_hidden) + "-fea-"\
			 + str(FLAGS.feature) 
	file_tail += "-" + str(FLAGS.feature_emb_size) if FLAGS.feature_emb_size != 25 else ""
	file_tail += "-epoch-" + str(FLAGS.epoch_step) 
	file_tail += "-ranemb" if FLAGS.ran_emb == 1 else ""
	return file_tail

def dynamic_rnn(sentence_num = 0,max_len = 54):
	train_data = Dataset(data_type = 'train')
	test_data = Dataset(data_type = 'test')

	x_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None]) #[FLAGS.batch_size,None]
	pos_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    	ner_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
	y_ = tf.placeholder(tf.int32, [None])
	mask = tf.placeholder(tf.int32,[None])
    	output_keep_prob = tf.placeholder(tf.float32)
	seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.abs(x_)), 1),tf.int32)
	tf.set_random_sees(1)
	#x:[batch_size,n_steps,n_input]
	with tf.device('/cpu:0'):
		embedding = pkl.load(open(FLAGS.embedding_path, 'r'))
		x = tf.nn.embedding_lookup(embedding, x_)
		biases = tf.get_variable("biases", [FLAGS.n_classes], tf.float32)
		if FLAGS.ran_emb == 1:
			random_emb = tf.get_variable("ran_emb", [WORD_NUM, FLAGS.word_dim], tf.float32)
		        ran_x = tf.nn.embedding_lookup(random_emb, x_)
			x = tf.concat(2,[x, ran_x])

	if FLAGS.feature == 1:
            	pos_emb = tf.get_variable("pos_emb", [POS_NUM, FLAGS.feature_emb_size], tf.float32)
            	ner_emb = tf.get_variable("ner_emb", [NER_NUM, FLAGS.feature_emb_size], tf.float32)
            	p = tf.nn.embedding_lookup(pos_emb, pos_)
            	n = tf.nn.embedding_lookup(ner_emb, ner_)
            	x = tf.concat(2,[x, p, n])


	if FLAGS.BiLSTM == 0:
		with tf.device('/gpu:2'):
			weight_shape = [FLAGS.n_hidden,FLAGS.n_classes]
			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden,state_is_tuple=True,activation=tf.nn.relu)
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=1-FLAGS.dropout)
			outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x,sequence_length = seq_len, dtype=tf.float32)
			outputs = tf.reshape(outputs,[-1,FLAGS.n_hidden])
	else:
		with tf.device('/cpu:0'):
			weight_shape = [2*FLAGS.n_hidden,FLAGS.n_classes]
			lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden,state_is_tuple=True,activation=tf.nn.relu)		
			lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden,state_is_tuple=True,activation=tf.nn.relu)
			#output is a tuple e.g. ([batch_size, n_step, n_hidden],[batch_size, n_step, n_hidden]),t[0]:numpy.ndarray
			output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x, seq_len, dtype=tf.float32)
			out_fw = tf.convert_to_tensor(output[0], tf.float32)
			out_bw = tf.convert_to_tensor(output[1], tf.float32)
			outputs = tf.concat(2, [out_fw,out_bw]) 
			outputs = tf.reshape(outputs,[-1,2*FLAGS.n_hidden])
	
	# Get lstm cell output
	with tf.device('/gpu:2'):
		if not FLAGS.L2:			
			weights = tf.get_variable("weights",weight_shape, tf.float32)
		else:
			weights = tf.get_variable("weights", weight_shape, tf.float32,
					initializer = tf.truncated_normal_initializer(stddev=0.01))
		logits = tf.matmul(outputs, weights) + biases
		if not FLAGS.CRF:
			cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_)*tf.cast(mask,tf.float32)) + BETA_REGUL * tf.nn.l2_loss(weights)
		else:
			unary_score = tf.reshape(logits,[-1,max_len,FLAGS.n_classes])
			tag_ind = tf.reshape(y_,[FLAGS.batch_size,max_len])
			with tf.device('/cpu:0'):
				log_likelihood,transition_params = tf.contrib.crf.crf_log_likelihood(unary_score,tag_ind,seq_len)
			cost = tf.reduce_mean(-log_likelihood) + BETA_REGUL* tf.nn.l2_loss(weights)
			
	optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(cost)
	correct_prediction = tf.nn.in_top_k(logits, y_, 1)
	values, indices = tf.nn.top_k(logits, 1)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	correct = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))


	file_tail = get_name_tail()
    	if FLAGS.PRF == 0:
		with tf.Session(config = gpu_config()) as sess:
			saver = tf.train.Saver()
			best_acc = 0
			sess.run(tf.initialize_all_variables())
			logConfig(FLAGS.log_path + file_tail)
			for step in range(FLAGS.epoch_step):
				for i in range(0, train_data.sentence_num):
				#for i in range(0, 300):
					batch_x, batch_pos, batch_ner, batch_y = train_data.next_batch()
					x_pad,y_pad,pos_pad,ner_pad,mask_feed = padding(batch_x,batch_y,batch_pos,batch_ner)
					out,_ = sess.run([outputs,optimizer], feed_dict = {x_:x_pad, y_:y_pad,mask:mask_feed,\
					pos_:pos_pad, ner_:ner_pad, output_keep_prob:1-FLAGS.dropout})
				num = 0
				cor_num = 0
				for i in range(0, test_data.sentence_num):
					test_x, test_pos, test_ner, test_y = test_data.next_batch()
					x_pad,y_pad,pos_pad,ner_pad,mask_feed = padding(test_x,test_y,test_pos,test_ner)
					if not FLAGS.CRF:
						num += len(test_y[0])
						seq_length,correct_num = sess.run([seq_len,correct],feed_dict = {x_:x_pad,\
						 y_:y_pad,mask:mask_feed,pos_:pos_pad, ner_:ner_pad, output_keep_prob:1})
						cor_num += correct_num
					else:
						seq_length,correct_num,tran_matrix,score = sess.run([seq_len,correct,\
							transition_params,unary_score],feed_dict = {x_:x_pad, y_:y_pad,\
							 mask:mask_feed, pos_:pos_pad, ner_:ner_pad, output_keep_prob:1})
						cor_label, label_num, _ = crf_evaluate(seq_length,tran_matrix,score,y_pad)
						cor_num += cor_label
						num += label_num
				test_accuracy = 1.0*cor_num/num
				if test_accuracy >= best_acc:
					saver.save(sess,FLAGS.saver_path + file_tail)
					best_acc = test_accuracy
				#print "step %d , test_accuracy: %g" % (step,test_accuracy)
				logging.info("step %d , test_accuracy: %g" % (step,test_accuracy))
	else:
		with tf.Session(config = gpu_config()) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, FLAGS.saver_path + file_tail)
			label_dict = pkl.load(open(FLAGS.label_dict_path,'r'))
			word_dict = pkl.load(open(FLAGS.word_dict_path,'r'))
			out = open(FLAGS.output_path + file_tail ,'w')
			num = 0
			cor_num = 0
			for i in range(0, test_data.sentence_num):
				test_x, test_pos, test_ner, test_y = test_data.next_batch()
				x_pad,y_pad,pos_pad,ner_pad,mask_feed = padding(test_x,test_y,test_pos,test_ner)
				if not FLAGS.CRF:
					num += len(test_y[0])
					ind,correct_num = sess.run([indices,correct],feed_dict = {x_:x_pad,\
					 y_:y_pad,mask:mask_feed,pos_:pos_pad, ner_:ner_pad, output_keep_prob:1})
					cor_num += correct_num
				else:
					seq_length,correct_num,tran_matrix,score = sess.run([seq_len,correct,\
						transition_params,unary_score],feed_dict = {x_:x_pad, y_:y_pad,\
						 mask:mask_feed, pos_:pos_pad, ner_:ner_pad, output_keep_prob:1})
					cor_label_num, label_num, ind = crf_evaluate(seq_length,tran_matrix,score,y_pad)
					cor_num += cor_label_num
					num += label_num
				for j in range(0,len(test_y[0])):
					out.write(word_dict[test_x[0][j]] + '\t' + label_dict[test_y[0][j]] + '\t' + label_dict[ind[j][0]] + '\n')
				out.write('\n')
				out.flush()

			print "cor_num",cor_num,"num=",num,"test_accuracy: %g" % (1.0*cor_num/num)
			out.close()


def main(_):
	dynamic_rnn()

if __name__ == '__main__':
	tf.app.run()
