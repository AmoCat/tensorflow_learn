#!usr/bin/env python
#coding:utf-8

import os
import sys
import tensorflow as tf
import numpy as np
import cPickle as pkl

class Dataset(object):
	def __init__(self, data_type = 'train', embedding_size = 345823, word_dim = 300, batch_size = 1):
		self.word_dim = word_dim
		self.data_type = data_type
		self.embedding_size = embedding_size
		self.words, self.pos, self.ner, self.labels,self.ran_words, self.sentence_num = self.init_data(self.data_type)
		self.batch_size = batch_size
		self.completed_epoches = 0
		self._index_in_epoch = 0

	def init_data(self, data_name):
        	datas = pkl.load(open(data_name + '_label', 'r'))
		ran_word_to_id = pkl.load(open('./dict/ran_word_to_id_dict', 'r'))
		word_to_id = pkl.load(open('./dict/minimized_word_to_id_dict', 'r'))
		pos_to_id = pkl.load(open('./dict/pos_to_id_dict', 'r'))
        	ner_to_id = pkl.load(open('./dict/ner_to_id_dict', 'r'))
        	label_to_id = pkl.load(open('./dict/label_to_id_dict', 'r'))
		
		words = list()
        	pos = list()
        	ner = list()
		labels = list()
		ran_words = list()
		for data in datas:
			ran_words.append([ran_word_to_id[t[0]] if ran_word_to_id.has_key(t[0]) \
							else ran_word_to_id['UNK'] for t in data])
            		words.append([word_to_id[t[0]] if word_to_id.has_key(t[0]) else word_to_id['UNK'] for t in data])
			pos.append([pos_to_id[t[1]] for t in data])
            		ner.append([ner_to_id[t[2]] for t in data])
            		labels.append([label_to_id[t[-1]] for t in data])
		#w = np.ndarray((len(words),), dtype = np.int32)
		#p = np.ndarray(()))
		#perm = np.arange(len(words))
		#np.random.shuffle(perm)
		#words = words[perm]
		#labels = labels[perm]
		return np.array(words), pos, ner, np.array(labels), np.array(ran_words), len(words)

	def next_batch(self):
		start = self._index_in_epoch
		self._index_in_epoch += self.batch_size
		if self._index_in_epoch >= self.sentence_num:
			self._index_in_epoch = 0
			end = self.sentence_num
			#perm = np.arange(self.sentence_num)
			#np.random.shuffle(perm)
			#self.words = self.words[perm]
			#self.labels = self.labels[perm]
		else:
			end = self._index_in_epoch
		return self.ran_words[start:end], self.words[start:end], self.pos[start:end], self.ner[start:end], self.labels[start:end]

	def get_datas(self):
		return self.words, self.labels
	
	def test(self):
		print self.words.shape,self.labels.shape	

if __name__ == '__main__':
	#data = Dataset('train')
	test = Dataset('test')
	test.next_batch()
	print test.sentence_num
