#!usr/bin/env python
#coding:utf-8

import os
import sys
import cPickle as pkl
import numpy as np
import collections
from collections import Counter
import tensorflow as tf

FILE_PATH = os.path.dirname(__file__)
EMBEDDING_SIZE = 345823
WORD_DIM = 300

'''
write[[(word,label),……],[],……]to './name_label'
e.g. word,pos,ner,label -> (word,label)
'''
def helper(name):
	in_path = os.path.join(FILE_PATH,name)
	out_path = os.path.join(FILE_PATH,name+'_label')
	datas = open(in_path,'r').read().strip().split('\n\n')
	out_l = []
	for data in datas:
		lines = data.split('\n')
		l = []
		for line in lines:
			words = line.split('\t')
			l.append((words[0],words[-1]))
		out_l.append(l)
	pkl.dump(out_l,open(out_path,'w'))

'''
read word and embedding,then generate word-to-id dict and id-to-word dict
use pkl to store dicts and embeddings
'''
def gen_word_id_dict(file_path,word_to_id_name = 'word_to_id_dict',\
	id_to_word_name = 'id_to_word_dict', embedding_name = 'embeddings'):
	datas = open(file_path,'r').read().strip().split('\n')
	embeddings = np.ndarray([EMBEDDING_SIZE,WORD_DIM],dtype = np.float32)
	word_to_id = dict()
	
	unk_embedding = np.random.rand(WORD_DIM)
	for line in datas:
		words = line.strip().split(' ')
		i = len(word_to_id)
		word_to_id[words[0]] = i
		for j in range(1,len(words)-1):
			embeddings[i][j-1] = words[j]
	word_to_id['UNK'] = 0
	for i in range(0,WORD_DIM-1):
		embeddings[0][i] = unk_embedding[i]
	id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
	pkl.dump(word_to_id,open(word_to_id_name, 'w'))
	pkl.dump(id_to_word,open(id_to_word_name, 'w'))
	pkl.dump(embeddings,open(embedding_name, 'w'))

def gen_label_id_dict(train_name = 'train_label',test_name = 'test_label',\
	 label_to_id_name = 'label_to_id_dict', id_to_label_name = 'id_to_label_dict'):
	train_data = pkl.load(open(train_name, 'r'))
	test_data = pkl.load(open(test_name, 'r'))

	#get label list
	label_list = list()
	for l in train_data:
		label_list.extend([t[1] for t in l])
	for l in test_data:
		label_list.extend(t[1] for t in l)

	#delete the duplicates in label list
	count = [['UNK',-1]]
	count.extend(collections.Counter(label_list).most_common())

	#convert label to id
	label_to_id = dict()
	for label, _ in count:
		label_to_id[label] = len(label_to_id)
	id_to_label = dict(zip(label_to_id.values(), label_to_id.keys()))
	pkl.dump(label_to_id, open(label_to_id_name, 'w'))
	pkl.dump(id_to_label, open(id_to_label_name, 'w'))


'''
test if file 'train_label' and 'test_label' is ready
'''
def test_label_data(name):
	path = out_path = os.path.join(FILE_PATH,name+'_label')
	datas = pkl.load(open(path,'r'))
	for l in datas:
		print l[0][0],l[0][1]
		return

def test_word_to_id_dict(name):
	dict = pkl.load(open(name, 'r'))
	print len(dict)

if __name__ == '__main__':
	gen_word_id_dict('../embedding/embedding/baike-300.vec.txt')
	#gen_label_id_dict()
	test_word_to_id_dict('word_to_id_dict')
