#!usr/bin/env python
#coding:utf-8

import os
import sys
import tensorflow as tf
import numpy as np
import cPickle as pkl

#np.random.seed(5)
#tf.set_random_seed(5)

class Dataset(object):
    def __init__(self, data_type = 'train',max_len = 54, embedding_size = 345823, word_dim = 300, batch_size = 16,max_path_num = 3):
        np.random.seed(1)
        tf.set_random_seed(1)
        self.word_dim = word_dim
        self.max_len = max_len
        self.max_path_num = max_path_num
        self.data_type = data_type
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.data_prefix = './data/'
        self.words,self.ran_words, self.real_sentence_num, self.sentence_num = self.init_data(self.data_type)
        self.completed_epoches = 0
        self._index_in_epoch = 0
        #self.batch_num = (self.sentence_num-1)/self.batch_size + 1
        self.batch_num = self.sentence_num/self.batch_size

    def init_path(self, add_sen_num, data_type, path_type = 'sdp', file_prefix = './data/path_pkl/', file_name = 'path_to_root'):
        file_path = file_prefix+data_type+'_'+path_type+'_1_'+file_name
        relation_path = file_prefix+data_type+'_'+path_type+'_idrelation_1_'+file_name
        path = pkl.load(open(file_path, 'r'))
        for i in range(0, add_sen_num):
            path.append([[1]])
        return np.array(path)

    def init_sdp_path(self, add_sen_num, mode, file_prefix = './data/path_pkl/',file_name = "all_path_to_root"):
        paths = pkl.load(open(file_prefix + mode + '_sdp_' + file_name,'r'))
        relations = pkl.load(open(file_prefix + mode + '_sdp_idrelation_' + file_name,'r'))
        for i in range(len(paths)):
            sen = paths[i]
            for j in range(len(sen)):
                w = sen[j]
                if len(w) > self.max_path_num:#去掉多于最大路径数值的路径
                    for k in range(len(w)-self.max_path_num):
                        paths[i][j].pop()
                        relations[i][j].pop()
                    str = "data[%d][%d] 路径数大于%d,修正后为%d"%(i,j,self.max_path_num,len(paths[i][j]))
                    continue
                pad_path_num = self.max_path_num - len(w)
                for k in range(pad_path_num):
                    paths[i][j].append([1])#补一个路径
                    relations[i][j].append([0])
        for i in range(add_sen_num):
            paths.append([[[1]]])
            relations.append([[[0]]])
        #paths = np.reshape(np.array(paths),[self.batch_size*self.max_len,-1])
        #relations = np.reshape(np.array(relations),[self.batch_size*self.max_len,-1])
        return np.array(paths),np.array(relations)

    def init_fea(self, datas, f, add_sen_num, name = './dict/label',padding_num = 0):
        if f!=4:
            fea_to_id = pkl.load(open(name + '_to_id_dict', 'r'))
        fea = list()
        for data in datas:
            if f!=4:
                l = []
                for t in data:
                    l.append(fea_to_id[t[f]] if fea_to_id.has_key(t[f]) else fea_to_id['unk'])
                fea.append(l)
            else:
                fea.append([t[f] for t in data])
        for i in range(0, add_sen_num):
            fea.append([padding_num])
        return  np.array(fea)

    def init_data(self, data_name):
        datas = pkl.load(open('./dict/' + data_name + '_label', 'r'))
        ran_word_to_id = pkl.load(open('./dict/ran_word_to_id_dict', 'r'))
        word_to_id = pkl.load(open('./dict/minimized_word_to_id_dict', 'r'))
        words = list()
        ran_words = list()
        for data in datas:
            ran_words.append([ran_word_to_id[t[0]] if ran_word_to_id.has_key(t[0]) \
                            else ran_word_to_id['UNK'] for t in data])
            words.append([word_to_id[t[0]] if word_to_id.has_key(t[0]) else word_to_id['UNK'] for t in data])
        sentence_num = len(words)
        print "sentence num:",len(words),'\n'
        pad_sen_num = 0
        if sentence_num % self.batch_size != 0:
            pad_sen_num = self.batch_size - (sentence_num % self.batch_size)
        padding = [0 for i in range(0, self.max_len)]
        for i in range(0, pad_sen_num):
            ran_words.append(padding)
            words.append(padding)
        fea_dict = {1:'pos',2:'ner',3:'sdp',4:'cur_sdp_father',-1:'label'}
        #fea_dict = {1:'pos',2:'ner',3:'sdp',-1:'label'}

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
            elif v == 'cur_sdp_father':
                self.cur_sdp_father = self.init_fea(datas,i,pad_sen_num,name,padding_num = 1)
        self.dp_path = self.init_path(pad_sen_num, data_name, path_type = 'sdp')
        self.sdp_path,self.sdp_relations = self.init_sdp_path(pad_sen_num, data_name)
        return np.array(words), np.array(ran_words), sentence_num, sentence_num + pad_sen_num

    def next_batch(self, max_seq = 54):
        start = self._index_in_epoch
        self._index_in_epoch += self.batch_size
        if self._index_in_epoch >= self.sentence_num:
            self._index_in_epoch = 0
            end = self.sentence_num
            if self.data_type == 'train':
                perm = np.arange(self.sentence_num)
                np.random.shuffle(perm)
                self.words = self.words[perm]
                self.pos = self.pos[perm]
                self.ner = self.ner[perm]
                self.sdp = self.sdp[perm]
                self.labels = self.labels[perm]
                self.sdp_path = self.sdp_path[perm]
                self.dp_path = self.dp_path[perm]
                self.sdp_relations = self.sdp_relations[perm]
                self.cur_sdp_father = self.cur_sdp_father[perm]
        else:
            end = self._index_in_epoch
        return self.ran_words[start:end], self.words[start:end], self.pos[start:end], \
                self.ner[start:end], self.sdp[start:end], self.labels[start:end], self.dp_path[start:end],\
                self.cur_sdp_father[start:end],self.sdp_path[start:end],self.sdp_relations[start:end]

    def get_datas(self):
        return self.words, self.labels

    def test(self):
        print self.words.shape,self.labels.shape

if __name__ == '__main__':
    #data = Dataset('train'):w
    test = Dataset('test')
    _,_,_,_,_,_,path,father = test.next_batch()
    print path[0]
    print "after padding,there are ", test.sentence_num, "sentences"
