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

flags.DEFINE_integer('ran_word_num',1976,'ran_word_num')
flags.DEFINE_integer('CRF',1,'CRF layer')
flags.DEFINE_integer('batch_size',16,'batch_size')
flags.DEFINE_integer('n_hidden',128,'hidden units')
flags.DEFINE_integer('epoch_step',40,'nums of epochs')
flags.DEFINE_integer('epoch_size',6000,'batchs of each epoch')
flags.DEFINE_integer('n_classes',18,'nums of classes')
flags.DEFINE_integer('emb_size',345823,'embedding size')
flags.DEFINE_integer('word_dim',300,'word dim')
flags.DEFINE_integer('PRF',0,'calculate PRF')
flags.DEFINE_integer('L2',1,'add L2 regularizer')
flags.DEFINE_integer('add_sdp',1,'add sdp feature')
flags.DEFINE_integer('add_sdp_in_crf',1,'add sdp in crf layer')
flags.DEFINE_integer('feature',1,'add pos and ner feature')
flags.DEFINE_integer('BiLSTM',1,'is bi-directional LSTM or not')
flags.DEFINE_integer('ran_emb',0,'add random variable embedding in training process')
flags.DEFINE_integer('pos_emb_size',50,'pos_embedding_size')
flags.DEFINE_integer('ner_emb_size',50,'ner_embedding_size')
flags.DEFINE_integer('sdp_emb_size',50,'sdp_embedding_size')
flags.DEFINE_integer('feature_emb_size',50,'pos and ner embedding size')
flags.DEFINE_float('learning_rate',1e-3,'learning rate')
flags.DEFINE_float('dropout',0,'dropout')
flags.DEFINE_float('beta',0.01,'beta_regul')
flags.DEFINE_string('data_path',None,'data path')
flags.DEFINE_string('embedding_path','./minimized_embeddings','embedding_path')
flags.DEFINE_string('saver_path','./model/model-','saver_path')
flags.DEFINE_string('output_path','./output/','prediction_output_path')
flags.DEFINE_string('label_dict_path','./dict/id_to_label_dict','id_to_label_dict_path')
flags.DEFINE_string('word_dict_path','./dict/minimized_id_to_word_dict','id_to_word_dict_path')
flags.DEFINE_string('log_path','./log/','log_path')

POS_NUM = 23
NER_NUM = 9
SDP_NUM = 107
#WORD_NUM = 345823
WORD_NUM = 1789
BETA_REGUL = 0.01


embedding = pkl.load(open(FLAGS.embedding_path, 'r'))
label_dict = pkl.load(open(FLAGS.label_dict_path,'r'))
word_dict = pkl.load(open(FLAGS.word_dict_path,'r'))

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
#		for x in y[i]:
#			print label_dict[x]," "
#		print "\n"
#		for label in viterbi_sequence_:
#			print label_dict[label]," "
		ind.extend([t] for t in viterbi_sequence)
		#print type(viterbi_sequence),"viterbi_seq:",viterbi_sequence_
		#print "\n","y:",y_
		#print "correct_num:",correct_num
	return correct_num,label_num,ind

def get_name_tail():
	file_tail = ""
	file_tail += "CRF" if FLAGS.CRF == 1 else ""
	file_tail += "-sdp" if FLAGS.add_sdp else ""
	file_tail += "-crfsdp" if FLAGS.add_sdp_in_crf else ""
	file_tail += "-L2" + str(FLAGS.beta) if FLAGS.L2 == 1 else ""
	file_tail += "-BILSTM" + str(FLAGS.BiLSTM) + "-h" + str(FLAGS.n_hidden) + "-fea-"\
			 + str(FLAGS.feature) 
	file_tail += "-" + str(FLAGS.feature_emb_size) if FLAGS.feature_emb_size != 25 else ""
	file_tail += "-epoch-" + str(FLAGS.epoch_step) 
	file_tail += "-ranemb" if FLAGS.ran_emb == 1 else ""
	file_tail += "-dropout" + str(FLAGS.dropout) if FLAGS.dropout != 0 else ""
	file_tail += "-b" + str(FLAGS.batch_size) if FLAGS.batch_size !=1 else ""
	return file_tail

def get_pad(data):
	random_x,batch_x, batch_pos, batch_ner,batch_sdp, batch_y = data.next_batch()
	ran_x_pad = padding_fea(random_x)
	sdp_pad = padding_fea(batch_sdp)
	x_pad,y_pad,pos_pad,ner_pad,mask_feed = padding(batch_x,batch_y,batch_pos,batch_ner)
	return batch_x,batch_y,x_pad, ran_x_pad, y_pad, mask_feed, pos_pad, ner_pad, sdp_pad

def dynamic_rnn(sentence_num = 0,max_len = 54):
	train_data = Dataset(data_type = 'train', batch_size = FLAGS.batch_size)
	test_data = Dataset(data_type = 'test', batch_size = FLAGS.batch_size)

	x_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None]) #[FLAGS.batch_size,None]
	x_ran = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
	pos_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    	ner_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
	sdp_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
	y_ = tf.placeholder(tf.int32, [None])
	mask = tf.placeholder(tf.int32,[None])
	out_keep_prob = tf.placeholder(tf.float32)
	seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.abs(x_)), 1),tf.int32)
	tf.set_random_seed(1)
	#x:[batch_size,n_steps,n_input]
	with tf.device('/cpu:0'):
		#embedding = pkl.load(open(FLAGS.embedding_path, 'r'))
		x = tf.nn.embedding_lookup(embedding, x_)
		biases = tf.get_variable("biases", [FLAGS.n_classes], tf.float32)
		if FLAGS.ran_emb == 1:
			random_emb = tf.get_variable("ran_emb", [FLAGS.ran_word_num, FLAGS.word_dim], tf.float32)
		        ran_x = tf.nn.embedding_lookup(random_emb, x_ran)
			x = tf.concat(2,[x, ran_x])

	if FLAGS.feature == 1:
            	pos_emb = tf.get_variable("pos_emb", [POS_NUM, FLAGS.feature_emb_size], tf.float32)
            	ner_emb = tf.get_variable("ner_emb", [NER_NUM, FLAGS.feature_emb_size], tf.float32)
            	p = tf.nn.embedding_lookup(pos_emb, pos_)
            	n = tf.nn.embedding_lookup(ner_emb, ner_)
            	x = tf.concat(2,[x, p, n])
		if FLAGS.add_sdp == 1:
			sdp_emb = tf.get_variable('sdp_emb', [SDP_NUM, FLAGS.feature_emb_size], tf.float32)
			sdp = tf.nn.embedding_lookup(sdp_emb, sdp_)
			x = tf.concat(2, [x, sdp])

	if FLAGS.BiLSTM == 0:
		with tf.device('/gpu:2'):
			weight_shape = [FLAGS.n_hidden,FLAGS.n_classes]
			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden,state_is_tuple=True,activation=tf.nn.relu)
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=out_keep_prob)
			outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x,sequence_length = seq_len, dtype=tf.float32)
			outputs = tf.reshape(outputs,[-1,FLAGS.n_hidden])
	else:
		with tf.device('/cpu:0'):
			weight_shape = [2*FLAGS.n_hidden,FLAGS.n_classes]
			lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden,state_is_tuple=True,activation=tf.nn.relu)		
			lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw,output_keep_prob=out_keep_prob)
			lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden,state_is_tuple=True,activation=tf.nn.relu)
			lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw,output_keep_prob=out_keep_prob)
			#output is a tuple e.g. ([batch_size, n_step, n_hidden],[batch_size, n_step, n_hidden]),t[0]:numpy.ndarray
			output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x, seq_len, dtype=tf.float32)
			out_fw = tf.convert_to_tensor(output[0], tf.float32)
			out_bw = tf.convert_to_tensor(output[1], tf.float32)
			outputs = tf.concat(2, [out_fw,out_bw]) 
			outputs = tf.reshape(outputs,[-1,2*FLAGS.n_hidden])
	
	# Get lstm cell output
	with tf.device('/gpu:2'):
		weights = tf.get_variable("weights", weight_shape, tf.float32,
					initializer = tf.truncated_normal_initializer(stddev=0.01))
		if FLAGS.add_sdp_in_crf:
			crf_sdp_embedding = tf.get_variable("crf_sdp",[SDP_NUM,FLAGS.feature_emb_size], tf.float32)
			crf_sdp = tf.nn.embedding_lookup(crf_sdp_embedding, sdp_)
			crf_sdp = tf.reshape(crf_sdp,[-1,FLAGS.feature_emb_size])
			outputs = tf.concat(1,[outputs, crf_sdp])
			crf_weights = tf.get_variable("crf_weights", [FLAGS.feature_emb_size, FLAGS.n_classes],\
				tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.01))
			weights = tf.concat(0,[weights, crf_weights])
		logits = tf.matmul(outputs, weights) + biases
		if not FLAGS.CRF:
			cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_)*tf.cast(mask,tf.float32))
		else:
			unary_score = tf.reshape(logits,[-1,max_len,FLAGS.n_classes])
			tag_ind = tf.reshape(y_,[FLAGS.batch_size,max_len])
			with tf.device('/cpu:0'):
				log_likelihood,transition_params = tf.contrib.crf.crf_log_likelihood(unary_score,tag_ind,seq_len)
			cost = tf.reduce_mean(-log_likelihood)
		if FLAGS.L2:
			cost += FLAGS.beta * tf.nn.l2_loss(weights)
	optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(cost)
	correct_prediction = tf.nn.in_top_k(logits, y_, 1)
	values, indices = tf.nn.top_k(logits, 1)
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	correct = tf.reduce_sum(tf.cast(correct_prediction,tf.float32)* tf.cast(mask, tf.float32))


	file_tail = get_name_tail()
    	if FLAGS.PRF == 0:
		with tf.Session(config = gpu_config()) as sess:
			saver = tf.train.Saver()
			best_acc = 0
			sess.run(tf.initialize_all_variables())
			logConfig(FLAGS.log_path + file_tail)
			for step in range(FLAGS.epoch_step):
				ave_cost = 0
				for i in range(0, train_data.batch_num):
					_,_,x_pad, ran_x_pad, y_pad, mask_feed, pos_pad, ner_pad, sdp_pad = get_pad(train_data)
					loss,out,_ = sess.run([cost,outputs,optimizer],\
						feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,\
						mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1-FLAGS.dropout})
					ave_cost += loss
				logging.info("step %d, training  loss : %g" % (step, ave_cost/train_data.batch_num))
				num = 0
				cor_num = 0
				ave_cost = 0
				for i in range(0, test_data.batch_num):
					test_x,test_y,x_pad, ran_x_pad, y_pad, mask_feed, pos_pad, ner_pad, sdp_pad = get_pad(test_data)
					if not FLAGS.CRF:
						for yi in test_y:
							num += len(yi)
						loss,seq_length,correct_num = sess.run([cost,seq_len,correct],\
							feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,\
							mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1})
						cor_num += correct_num
					else:
						loss,seq_length,correct_num,tran_matrix,score = sess.run([cost,seq_len,correct,\
							transition_params,unary_score],\
							feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,\
							mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1})
						#for j in range(0,seq_length[0]):
						#	print word_dict[test_x[0][j]] , " "
						cor_label, label_num, _ = crf_evaluate(seq_length,tran_matrix,score,y_pad)
						cor_num += cor_label
						num += label_num
					ave_cost += loss
				test_accuracy = 1.0*cor_num/num
				if test_accuracy >= best_acc:
					saver.save(sess,FLAGS.saver_path + file_tail)
					best_acc = test_accuracy
				#print "step %d , test_accuracy: %g" % (step,test_accuracy)
				logging.info("step %d , test_loss: %g,test_accuracy: %g" % (step,ave_cost/test_data.batch_num,test_accuracy))
	else:
		with tf.Session(config = gpu_config()) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, FLAGS.saver_path + file_tail)
			#label_dict = pkl.load(open(FLAGS.label_dict_path,'r'))
			#word_dict = pkl.load(open(FLAGS.word_dict_path,'r'))
			out = open(FLAGS.output_path + file_tail ,'w')
			num = 0
			cor_num = 0
			ave_cost = 0
			for i in range(0, test_data.batch_num):
				test_x,test_y,x_pad, ran_x_pad, y_pad, mask_feed, pos_pad, ner_pad, sdp_pad = get_pad(test_data)
				if not FLAGS.CRF:
					for yi in test_y:
						num += len(yi)
					loss,seq_length,ind,correct_num = sess.run([cost,seq_len,indices,correct],\
						feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,\
						mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1})
					cor_num += correct_num
				else:
					loss,seq_length,correct_num,tran_matrix,score = sess.run([cost,seq_len,correct,\
						transition_params,unary_score],\
						feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,\
						mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1})
					cor_label_num, label_num, ind = crf_evaluate(seq_length,tran_matrix,score,y_pad)
					cor_num += cor_label_num
					num += label_num
				ave_cost += loss
				for k in range(0,FLAGS.batch_size):
					for j in range(0,seq_length[k]):
						out.write(word_dict[test_x[k][j]] + '\t' + label_dict[test_y[k][j]] + '\t' + label_dict[ind[k*max_len+j][0]] + '\n')
					out.write('\n')
				out.flush()

			print "test loss:%g,test_accuracy: %g" % (ave_cost/test_data.batch_num,1.0*cor_num/num)
			out.close()


def main(_):
	dynamic_rnn()

if __name__ == '__main__':
	tf.app.run()
