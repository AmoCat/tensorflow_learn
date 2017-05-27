#!usr/bin/env python
#coding:utf-8

import os
import sys
import tensorflow as tf
import cPickle as pkl
import numpy as np
import collections
from collections import Counter
import tensorflow as tf
from copy import deepcopy

DICT_PREFIX = "./dict/"
FILE_PATH = os.path.dirname(__file__) + 'dict/'
EMBEDDING_SIZE = 345823
WORD_DIM = 300
DATA_PATH = './data/'

hash = {1:"pos",2:"ner",-1:"label",3:"sdp"}

'''
write[[[word,pos,ner,label],……],[],……]to './name_label'
'''
def helper(name):
	in_path = os.path.join(DATA_PATH,name)
	out_path = os.path.join(FILE_PATH,name+'_label')
	datas = open(in_path,'r').read().strip().split('\n\n')
	out_l = []
	for data in datas:
		lines = data.split('\n')
		l = []
		for line in lines:
			words = line.split('\t')
			l.append([word for word in words])
		out_l.append(l)
	pkl.dump(out_l,open(out_path,'w'))

'''
read word and embedding,then generate word-to-id dict and id-to-word dict
use pkl to store dicts and embeddings
file_path is path of original embeddings
'''
def gen_word_id_dict(file_path = "./baike-300.vec.txt",word_to_id_name = './dict/word_to_id_dict',\
	id_to_word_name = './dict/id_to_word_dict', embedding_name = 'embeddings'):
	datas = open(file_path,'r').read().strip().split('\n')
	embeddings = np.ndarray([EMBEDDING_SIZE,WORD_DIM],dtype = np.float32)
	word_to_id = dict()
	
	unk_embedding = np.random.rand(WORD_DIM)
	for line in datas:
		words = line.strip().split(' ')
		i = len(word_to_id)
		word_to_id[words[0]] = i
		for j in range(1,len(words)):
			embeddings[i][j-1] = words[j]
	word_to_id['UNK'] = 0
	for i in range(0,WORD_DIM):
		embeddings[0][i] = unk_embedding[i]
	id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
	pkl.dump(word_to_id,open(word_to_id_name, 'w'))
	pkl.dump(id_to_word,open(id_to_word_name, 'w'))
	pkl.dump(embeddings,open(embedding_name, 'w'))


def gen_ran_word2int():
	word2int = pkl.load(open("./dict/minimized_word_to_id_dict", 'r'))
	train = pkl.load(open("train_label", 'r'))
#	test = pkl.load(open("./data/test_label", 'r'))

	words = []
	for sen in train:
		words.extend([l[0] for l in sen])
	##minimized_id_to_words start from
	ran_word2int = word2int
	for word in words:
		if not word2int.has_key(word):
			ran_word2int[word] = len(ran_word2int) + 1

	ran_int2word =  dict(zip(ran_word2int.values(), ran_word2int.keys()))
	pkl.dump(ran_word2int, open("./dict/ran_word_to_id_dict", 'w'))
	pkl.dump(ran_int2word, open("./dict/ran_id_to_word_dict", 'w'))

'''
Get the embedding of the words contained in the training set and the test set;
save the embeddings in out_emb_path
'''
def minimize_words_size(embedding_path = "embeddings", train_path = "train_label", test_path = "test_label",\
		out_emb_path = "minimized_embeddings", word_to_id_path = "./dict/word_to_id_dict"):
	embeddings = pkl.load(open(embedding_path, 'r'))
	word_to_id_dict = pkl.load(open(word_to_id_path, 'r'))
	train_data = pkl.load(open(train_path, 'r'))
	test_data = pkl.load(open(test_path, 'r'))
	word_to_id_dict = pkl.load(open(word_to_id_path, 'r'))
	words = []
	for sen in train_data:
		words.extend([l[0] for l in sen])
	for sen in test_data:
		words.extend([l[0] for l in sen])
	words.append('UNK')

	new_emb = list()
	n_word_to_id_dict = dict()

	count = []
	count.extend(collections.Counter(words).most_common())
	
	first = []
	for j in range(0,WORD_DIM):
		first.append(embeddings[0][j])
	new_emb.append(first)
	for word,_ in count:
		# if embeddings has the word
		if word_to_id_dict.has_key(word):
			i = len(new_emb)
			id = word_to_id_dict[word]
			n_word_to_id_dict[word] = i
			e = list()
			for j in range(0,WORD_DIM):
				e.append(embeddings[id][j])
			new_emb.append(e)

	new_emb_nparray = np.array(new_emb)
	n_id_to_word_dict = dict(zip(n_word_to_id_dict.values(),n_word_to_id_dict.keys()))
	pkl.dump(new_emb_nparray, open("minimized_embeddings", 'w'))
	pkl.dump(n_id_to_word_dict, open("./dict/minimized_id_to_word_dict", 'w'))
	pkl.dump(n_word_to_id_dict, open("./dict/minimized_word_to_id_dict", 'w'))
	return len(n_id_to_word_dict)

def gen_sdp_path_relationid(mode = 'train', PREFIX = './data/path_pkl/', path_num = 'all'):
	in_tail = "_sdp_relation_"+path_num+"_path_to_root"
	out_tail = "_sdp_idrelation_"+path_num+"_path_to_root"
	sdp2int = pkl.load(open("./dict/sdp_to_id_dict",'r'))
	datas = pkl.load(open(PREFIX + mode + in_tail,'r'))
	id_datas = deepcopy(datas)
	for i in range(len(datas)):
		sen = datas[i]
		for j in range(len(sen)):
			w = sen[j]
			for k in range(len(w)):
				path = w[k]
				for q in range(len(path)):
					relation = path[q]
					if not sdp2int.has_key(relation):
						relation = 'unk'
					id_datas[i][j][k][q] = sdp2int[relation]
	pkl.dump(id_datas,open(PREFIX + mode + out_tail,'w'))

def gen_path_relationid(mode = 'train',data_type = 'dp', PREFIX = './data/path_pkl/', path_num = '1'):
	in_tail = "_" +data_type + "_relation_"+path_num+"_path_to_root"
	out_tail = "_" + data_type + "_idrelation_"+path_num+"_path_to_root"
	sdp2int = pkl.load(open("./dict/" + data_type + "_to_id_dict",'r'))
	datas = pkl.load(open(PREFIX + mode + in_tail,'r'))
	for i in range(len(datas)):
		sen = datas[i]
		for j in range(len(sen)):
			path = sen[j]
			for k in range(len(path)):
				relation = path[k]
				if not sdp2int.has_key(relation):
					relation = 'unk'
				datas[i][j][k] = sdp2int[relation]
	pkl.dump(datas,open(PREFIX + mode + out_tail,'w'))

'''
gen label,pos or ner feature hash,f is the index in file of datas[word,feature,label]
'''
def gen_label_id_dict(train_name = './dict/train_label',test_name = './dict/test_label',f = -1,\
	dev_name = './dict/dev_label',\
	label_to_id_name = 'label_to_id_dict', id_to_label_name = 'id_to_label_dict'):
	train_data = pkl.load(open(train_name, 'r'))
	test_data = pkl.load(open(test_name, 'r'))
	dev_data = pkl.load(open(dev_name, 'r'))

	#get label list
	label_list = list()
	for l in train_data:
		label_list.extend([t[f] for t in l])
	for l in test_data:
		label_list.extend(t[f] for t in l)
	for l in dev_data:
		label_list.extend(t[f] for t in l)

	#delete the duplicates in label list
	count = []
	count.extend(collections.Counter(label_list).most_common())

	#convert label to id
	label_to_id = dict()
	if f == 3 or f == 5:
		label_to_id['start'] = len(label_to_id)
	for label, _ in count:
		label_to_id[label] = len(label_to_id)
	if f >=1 and f <=5:
		label_to_id['unk'] = len(label_to_id)
	id_to_label = dict(zip(label_to_id.values(), label_to_id.keys()))
	print label_to_id_name, len(label_to_id)
	pkl.dump(label_to_id, open(label_to_id_name, 'w'))
	pkl.dump(id_to_label, open(id_to_label_name, 'w'))

def feature_hash(train_name = 'train_label',test_name = 'test_label'):
   	hash = {1:"pos",2:"ner",-1:"label",3:"sdp",5:"dp"}
    	for i,v in hash.items():
        	para1 = DICT_PREFIX + v + "_to_id_dict"
        	para2 =	DICT_PREFIX + "id_to_" + v + "_dict"
        	gen_label_id_dict(f = i,label_to_id_name = para1,id_to_label_name = para2)

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
	#helper("train")
	
	dict = pkl.load(open(name, 'r'))
	print len(dict)

if __name__ == '__main__':
	#words_num = minimize_words_size()
	#print "word_num:",words_num
	helper('train')
	helper('test')
	helper('dev')
	feature_hash()
	#gen_sdp_path_relationid(mode = 'train')
	#gen_sdp_path_relationid(mode = 'test')
	#gen_sdp_path_relationid(mode = 'dev')
	gen_path_relationid(mode = 'train')
	gen_path_relationid(mode = 'test')
	gen_path_relationid(mode = 'dev')
	#gen_word_id_dict(file_path = '../../embedding/embedding/baike-300.vec.txt')
	#feature_hash()
    	#gen_ran_word2int()
	#gen_label_id_dict()
	#test_word_to_id_dict(sys.argv[1])
