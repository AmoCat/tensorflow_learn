#!usr/bin/env python
#coding:utf-8

import os
import sys
import tensorflow as tf
import numpy as np
import cPickle as pkl

class Dataset(object):
	def __init__(self, data_type = 'train',max_len = 54, embedding_size = 345823, word_dim = 300, batch_size = 32):
		self.word_dim = word_dim
		self.max_len = max_len
		self.data_type = data_type
		self.embedding_size = embedding_size
		self.batch_size = batch_size
		self.words,self.ran_words, self.real_sentence_num, self.sentence_num = self.init_data(self.data_type)
		self.completed_epoches = 0
		self._index_in_epoch = 0
		#self.batch_num = (self.sentence_num-1)/self.batch_size + 1
		self.batch_num = self.sentence_num/self.batch_size

	def init_fea(self, datas, f, add_sen_num, name = './dict/label'):
		fea_to_id = pkl.load(open(name + '_to_id_dict', 'r'))
		fea = list()
		for data in datas:
			fea.append([fea_to_id[t[f]] for t in data])
		for i in range(0, add_sen_num):
			fea.append([0])
		return  np.array(fea)

	def init_data(self, data_name):
        	datas = pkl.load(open(data_name + '_label', 'r'))
		ran_word_to_id = pkl.load(open('./dict/ran_word_to_id_dict', 'r'))
		word_to_id = pkl.load(open('./dict/minimized_word_to_id_dict', 'r'))
	#	pos_to_id = pkl.load(open('./dict/pos_to_id_dict', 'r'))
        #	ner_to_id = pkl.load(open('./dict/ner_to_id_dict', 'r'))
        #	label_to_id = pkl.load(open('./dict/label_to_id_dict', 'r'))
	#	sdp_to_id = pkl.load(open('./dict/sdp_to_id_dict', 'r'))	
	
		words = list()
		ran_words = list()
        	#pos = list()
        	#ner = list()
		#labels = list()
		#sdp = list()
		for data in datas:
			ran_words.append([ran_word_to_id[t[0]] if ran_word_to_id.has_key(t[0]) \
							else ran_word_to_id['UNK'] for t in data])
            		words.append([word_to_id[t[0]] if word_to_id.has_key(t[0]) else word_to_id['UNK'] for t in data])
#			pos.append([pos_to_id[t[1]] for t in data])
#            		ner.append([ner_to_id[t[2]] for t in data])
#            		labels.append([label_to_id[t[-1]] for t in data]
#			sdp.appen([sdp_to_id[t[3]] for t in data]))
		sentence_num = len(words)
		print "sentence num:",len(words),'\n'
		pad_sen_num = 0
		if sentence_num % self.batch_size != 0:
			pad_sen_num = self.batch_size - (sentence_num % self.batch_size)
		padding = [0 for i in range(0, self.max_len)]
		for i in range(0, pad_sen_num):
			ran_words.append(padding)
			words.append(padding)
		fea_dict = {1:'pos',2:'ner',3:'sdp',-1:'label'}
		
		for i,v in fea_dict.items():
			name = './dict/' + v
			res = self.init_fea(datas,i,pad_sen_num,name)
			if v == 'pos':
				self.pos = res
			elif v == 'ner':
				self.ner = res
			elif v == 'sdp':
				self.sdp = res
			elif v == 'label':
				self.labels = res
		return np.array(words), np.array(ran_words), sentence_num, sentence_num + pad_sen_num

	def next_batch(self, max_seq = 54):
		start = self._index_in_epoch
		self._index_in_epoch += self.batch_size
		if self._index_in_epoch >= self.sentence_num:
			self._index_in_epoch = 0
			end = self.sentence_num
			perm = np.arange(self.sentence_num)
			np.random.shuffle(perm)
			self.words = self.words[perm]
			self.pos = self.pos[perm]
			self.ner = self.ner[perm]
			self.sdp = self.sdp[perm]
			self.labels = self.labels[perm]
		else:
			end = self._index_in_epoch
		return self.ran_words[start:end], self.words[start:end], self.pos[start:end], \
			self.ner[start:end], self.sdp[start:end], self.labels[start:end]

	def get_datas(self):
		return self.words, self.labels
	
	def test(self):
		print self.words.shape,self.labels.shape	

if __name__ == '__main__':
	#data = Dataset('train'):w

	test = Dataset('train')
	test.next_batch()
	print "after padding,there are ", test.sentence_num, "sentences"
