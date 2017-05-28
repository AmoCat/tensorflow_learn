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
import commands


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('ran_word_num',1976,'ran_word_num')
flags.DEFINE_integer('CRF',1,'CRF layer')
flags.DEFINE_integer('batch_size',16,'batch_size')
flags.DEFINE_integer('n_hidden',128,'hidden units')
flags.DEFINE_integer('encode_hidden',64, 'encode hidden units')
flags.DEFINE_integer('encode_dp_path',0, 'encode dp path,each word only one path to ROOT')
flags.DEFINE_integer('encode_sdp_path',0,'encode sdp path')
flags.DEFINE_integer('epoch_step',40,'nums of epochs')
flags.DEFINE_integer('epoch_size',6000,'batchs of each epoch')
flags.DEFINE_integer('n_classes',16,'nums of classes')
flags.DEFINE_integer('emb_size',345823,'embedding size')
flags.DEFINE_integer('word_dim',300,'word dim')
flags.DEFINE_integer('PRF',0,'calculate PRF')
flags.DEFINE_integer('L2',1,'add L2 regularizer')
flags.DEFINE_integer('add_sdp_anc',0,'add sdp anc')
flags.DEFINE_integer('add_sdp_anc_in_crf',0,'add sdp anc in crf layer')
flags.DEFINE_integer('add_sdp',0,'add sdp feature')
flags.DEFINE_integer('add_sdp_in_crf',0,'add sdp in crf layer')
flags.DEFINE_integer('feature',1,'add pos and ner feature')
flags.DEFINE_integer('BiLSTM',1,'is bi-directional LSTM or not')
flags.DEFINE_integer('ran_emb',0,'add random variable embedding in training process')
flags.DEFINE_integer('pos_emb_size',50,'pos_embedding_size')
flags.DEFINE_integer('ner_emb_size',50,'ner_embedding_size')
flags.DEFINE_integer('sdp_emb_size',50,'sdp_embedding_size')
flags.DEFINE_integer('sdp_label_in_path',0,'sdp_label_in_path')
flags.DEFINE_integer('dp_label_in_path',0,'dp_label_in_path')
flags.DEFINE_integer('feature_emb_size',50,'pos and ner embedding size')
flags.DEFINE_float('learning_rate',1e-3,'learning rate')
flags.DEFINE_float('dropout',0,'dropout')
flags.DEFINE_float('beta',0.0001,'beta_regul')
flags.DEFINE_float('encode_sdp_dropout',0.5, 'encode dropout')
flags.DEFINE_string('data_path',None,'data path')
flags.DEFINE_string('embedding_path','./minimized_embeddings','embedding_path')
flags.DEFINE_string('saver_path','./anc/model/model-','saver_path')
flags.DEFINE_string('output_path','./anc/output/','prediction_output_path')
flags.DEFINE_string('label_dict_path','./dict/id_to_label_dict','id_to_label_dict_path')
flags.DEFINE_string('word_dict_path','./dict/minimized_id_to_word_dict','id_to_word_dict_path')
flags.DEFINE_string('log_path','./anc/log/','log_path')
flags.DEFINE_string('PRF_path','./anc/PRFresult/','PRF_path')

SEED = 1
POS_NUM = 24
NER_NUM = 10
SDP_NUM = 108
DP_NUM = 16
#WORD_NUM = 345823
WORD_NUM = 1789
BETA_REGUL = 0.01
MAX_PATH_NUM = 3

padding_path_num = 0

embedding = pkl.load(open(FLAGS.embedding_path, 'r'))
label_dict = pkl.load(open(FLAGS.label_dict_path,'r'))
word_dict = pkl.load(open(FLAGS.word_dict_path,'r'))


class Encoder(object):
    def bilstm_encoder(self,father_index,batch_size,max_seq_len):
        with tf.variable_scope('encode_path_bilstm'):
            #anc_seq_len = tf.reduce_sum(tf.sign(father_mask),axis = 1)
            #father_index对于padding的word加了一个路径为[1]
            anc_seq_len = tf.reduce_sum(tf.sign(father_index),axis = 1)
            encode_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.encode_hidden,state_is_tuple = True, activation = tf.nn.relu)
            encode_cell_fw = tf.nn.rnn_cell.DropoutWrapper(encode_cell_fw, output_keep_prob = 1-FLAGS.encode_sdp_dropout, seed = SEED)
            encode_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.encode_hidden,state_is_tuple = True, activation = tf.nn.relu)
            encode_cell_bw = tf.nn.rnn_cell.DropoutWrapper(encode_cell_bw, output_keep_prob = 1-FLAGS.encode_sdp_dropout, seed = SEED)
    
            encode_outputs, _ = tf.nn.bidirectional_dynamic_rnn(encode_cell_fw, encode_cell_bw,\
                anc_feature, anc_seq_len, dtype = tf.float32)
    
            encode_output_fw,encode_output_bw = encode_outputs
            anc_batch_range = tf.range(FLAGS.batch_size*max_seq_len)
            fw_last_indices = tf.stack([anc_batch_range, anc_seq_len-1], axis = 1)
            zeros = tf.zeros([tf.shape(anc_seq_len)[0]],dtype = tf.int32)
            bw_last_indices = tf.stack([anc_batch_range, zeros], axis = 1)
    
            #[batch_size*max_len,anc_hidden]
            fw_encode_vec = tf.gather_nd(encode_output_fw, fw_last_indices)
            bw_encode_vec = tf.gather_nd(encode_output_bw, bw_last_indices)
    
            encode_vec = tf.concat(1,[fw_encode_vec,bw_encode_vec])
            encode_vec = tf.reshape(encode_vec, [FLAGS.batch_size,max_seq_len,2*FLAGS.encode_hidden])
            return encoder_vec

    def lstm_encoder(self,father_index,anc_feature,anc_seq_len = None):
        with tf.variable_scope('encode_path_lstm'):
            #anc_seq_len = tf.reduce_sum(tf.sign(father_mask),axis = 1)
            #father_index对于padding的word加了一个路径为[1]
            if anc_seq_len == None:
                anc_seq_len = tf.reduce_sum(tf.sign(father_index),axis = 1)
            encode_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.encode_hidden,state_is_tuple = True, activation = tf.nn.relu)
            encode_cell_fw = tf.nn.rnn_cell.DropoutWrapper(encode_cell_fw, output_keep_prob = 1-FLAGS.encode_sdp_dropout, seed = SEED)
    
            encode_outputs, _ = tf.nn.dynamic_rnn(encode_cell_fw,anc_feature, anc_seq_len, dtype = tf.float32)

            batch_range = tf.range(tf.shape(father_index)[0])
            fw_last_indices = tf.stack([batch_range, anc_seq_len-1], axis = 1)
            encode_vec = tf.gather_nd(encode_outputs, fw_last_indices)
            return encode_vec

def get_padding_path_num(sdp_father,sdp_hash_id):
    global padding_path_num
    padding_path_num = len(sdp_hash_id)-len(sdp_father)


def evaluate(best_F,file_tail,data_type):
    cmd = "perl ./conlleval.pl -d \"\\t\" < " + FLAGS.output_path + data_type + '/' +file_tail+ "tmp_out"
    status,output = commands.getstatusoutput(cmd)
    l = output.split('\n')
    if(l == None or len(l)<= 1):
        print "status:",status,"\noutput:",output
    res = l[1]
    F = float(l[1].split(' ')[-1])
    accuracy = l[1].split(' ')[1]
    logging.info("%s\n" % res)
    if F >= best_F:
        best_F = F
        f = open(FLAGS.PRF_path + '/' + data_type+'/'+file_tail, 'w')
        f.write(output)
        f.flush()
    return best_F, F, accuracy, F >= best_F


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
#       for x in y[i]:
#           print label_dict[x]," "
#       print "\n"
#       for label in viterbi_sequence_:
#           print label_dict[label]," "
        ind.extend([t] for t in viterbi_sequence)
        #print type(viterbi_sequence),"viterbi_seq:",viterbi_sequence_
        #print "\n","y:",y_
        #print "correct_num:",correct_num
    return correct_num,label_num,ind

def endChunk(s, lastType):
    if s == 'O' or s[0] == 'B':
        return True
    return s[2:] != lastType

'''
def calculate_fb1(batch_size, y, y_pred, seq_len, max_len):
    correctChunk = 0
    foundGuessed = 0
    foundCorrect = 0

    for i in range(0, batch_size):
        tag = [label_dict[id] for id in y[:seq_len[i]]]
        tag_pred = [label_dict[id] for id in y_pred[i*max_len:i*max_len + seq_len[i]]]
        FoundType = 'O'
        CorrectType = 'O'
        lastFoundIndex = -1
        lastCorrectIndex = -1
        ly = list()
        ly_pred = list()
        for j in range(0, seq_len[j]):
            a = tag[j]
            b = tag_pred[j]
            if endChunk(a, CorrectType) and CorrectType != 'O':
                ly.append((lastCorrectIndex,j,CorrectType))
                lastCorrectIndex = j
                CorrectType = 'O' if a == 'O' else a[2:]
                correctChunk += 1
            if endChunk(b, FoundType) and FoundType != 'O':
                ly_pred.append((lastFoundIndex,j,FoundType))
                lastFoundIndex = j
                FoundType = 'O' if b == 'O' else b[2:]
                foundGuesses += 1
        for item in ly:
            if item in ly_pred:
                foundCorrect += 1
        return correctChunk, foundGuessed, foundCorrect
'''
#def get_name_tail():
#    file_tail = ""
#    file_tail += "CRF" if FLAGS.CRF == 1 else ""
#    file_tail += "-sdp" if FLAGS.add_sdp else ""
#    file_tail += "-crfsdp" if FLAGS.add_sdp_in_crf else ""
#    file_tail += "-L2" + str(FLAGS.beta) if FLAGS.L2 == 1 else ""
#    file_tail += "-BILSTM" + str(FLAGS.BiLSTM) + "-h" + str(FLAGS.n_hidden) + "-fea-"\
#             + str(FLAGS.feature)
#    file_tail += "-" + str(FLAGS.feature_emb_size) if FLAGS.feature_emb_size != 25 else ""
#    file_tail += "-epoch-" + str(FLAGS.epoch_step)
#    file_tail += "-ranemb" if FLAGS.ran_emb == 1 else ""
#    file_tail += "-dropout" + str(FLAGS.dropout) if FLAGS.dropout != 0 else ""
#    file_tail += "-b" + str(FLAGS.batch_size) if FLAGS.batch_size !=1 else ""
#    return file_tail

def get_name_tail():
    file_tail = ""
    file_tail += "encode_dp_path-" if FLAGS.encode_dp_path == 1 else ""
    file_tail += "encode_sdp_path-" if FLAGS.encode_sdp_path == 1 else ""
    file_tail += "add_sdp_anc_in_crf-" if FLAGS.add_sdp_anc_in_crf == 1 else ""
    file_tail += "add_sdp_anc-" if FLAGS.add_sdp_anc == 1 else ""
    file_tail += "enchidden-" + str(FLAGS.encode_hidden)
    file_tail += "encsdpdropout" + str(FLAGS.encode_sdp_dropout)
    return file_tail 

def get_pad(data):
    random_x,batch_x, batch_pos, batch_ner,batch_sdp, batch_y, dp_path, batch_cur_sdp_father,\
     sdp_path,sdp_relation,dp_relation = data.next_batch()
    ran_x_pad = padding_fea(random_x)
    sdp_pad = padding_fea(batch_sdp)
    cur_sdp_father = padding_fea(batch_cur_sdp_father,padding_num = 1)
    x_pad,y_pad,pos_pad,ner_pad,mask_feed = padding(batch_x,batch_y,batch_pos,batch_ner)
    dp_path_pad= padding_path(dp_path)
    dp_path_mask = padding_path(dp_path,padding_num = 0)
    dp_relation_pad = padding_path(dp_relation, padding_num = 0)
    sdp_path_pad,sdppath_hash_id,sdp_seq_len,batch_range = padding_sdp_path(sdp_path,padding_num = 1)
    sdp_relation_pad,_,_,_ = padding_sdp_path(sdp_relation,padding_num = 0)
    return batch_x, batch_y, x_pad, ran_x_pad, y_pad, mask_feed, pos_pad, ner_pad, sdp_pad,\
     dp_path_pad,cur_sdp_father,sdp_path_pad,sdp_relation_pad,dp_relation_pad,sdppath_hash_id,\
     dp_path_mask,sdp_seq_len,batch_range

'''
ancestors_index size:[FLAGS.batch_size*max_seq_len, max_ancestors_len]
x size:[FLAGS.batch_size, max_seq_len]
return [batch_size*max_seq_len, max_ancestors_len]
ancestors_index start with 1
root index设为1,后序anc_seq_len每个减1去掉ROOT
'''
def anc_index_lookup(x_,ancestors_index,start_index = 1,batch_range = None):
    max_seq_len = tf.shape(x_)[1]
    batch_size = tf.shape(x_)[0]
    max_anc_len = tf.shape(ancestors_index)[1]

    anc_index_bias = tf.reshape(ancestors_index, [-1])-start_index 
    if batch_range == None:
        batch_range = tf.range(0,batch_size*max_seq_len*max_anc_len)/(max_anc_len*max_seq_len)
        hash_indices = batch_range*max_seq_len + anc_index_bias
    else:
        hash_indices = batch_range + anc_index_bias

    batch_len = max_seq_len*batch_size
    return batch_len,max_seq_len,max_anc_len,hash_indices

def anc_lookup(l, data, hash_id):
    with tf.device('/cpu:0'):
        data_copy = tf.reshape(data,[-1])
        return tf.reshape(tf.gather(data, hash_id), [l, -1])

def lookup_ancestors(x_,batch_len, pos_, ner_, hash_indices):
    anc_w = anc_lookup(batch_len, x_, hash_indices)
    anc_p = anc_lookup(batch_len, pos_, hash_indices)
    anc_n = anc_lookup(batch_len, ner_, hash_indices)
    return anc_w, anc_p, anc_n

def cur_father_lookup(x_, cur_father_index):
    batch_size = tf.shape(x_)[0]
    max_seq_len = tf.shape(x_)[1]
    w_table = tf.reshape(x_, [-1])
    anc_index_bias = tf.reshape(cur_father_index, [-1])-1
    batch_range = tf.range(0,batch_size*max_seq_len)/(max_seq_len)
    hash_indices = batch_range*max_seq_len + anc_index_bias
    cur_father_w = anc_lookup(batch_size, w_table, hash_indices)
    return cur_father_w

def dynamic_rnn(sentence_num = 0,max_len = 54):
    if FLAGS.PRF == 1:
        test_data = Dataset(data_type = 'test', batch_size = FLAGS.batch_size)
    else:
        train_data = Dataset(data_type = 'train', batch_size = FLAGS.batch_size)
        dev_data = Dataset(data_type = 'test', batch_size = FLAGS.batch_size)

    x_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None]) #[FLAGS.batch_size,None]
    x_ran = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    pos_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    ner_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    sdp_ = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    cur_sdp_father = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    y_ = tf.placeholder(tf.int32, [None])
    mask = tf.placeholder(tf.int32,[None])
   # ancestor_mask = tf.placeholder(tf.int32,[None])
    dp_father_index = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    dp_father_relation = tf.placeholder(tf.int32, [FLAGS.batch_size,None])
    dp_mask = tf.placeholder(tf.int32,[FLAGS.batch_size,None])
    sdp_father_index = tf.placeholder(tf.int32, [None,None])
    sdp_father_relation = tf.placeholder(tf.int32, [None,None])
    sdp_path_hash = tf.placeholder(tf.int32, [None])
    out_keep_prob = tf.placeholder(tf.float32)
    sdp_padding_path = tf.placeholder(tf.float32, [None,None])
    batch_range = tf.placeholder(tf.int32,[None])
    sdp_seq_len = tf.placeholder(tf.int32,[None])
    seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.abs(x_)), 1),tf.int32)
    #x:[batch_size,n_steps,n_input]
    with tf.device('/cpu:0'):
        #embedding = pkl.load(open(FLAGS.embedding_path, 'r'))
        x = tf.nn.embedding_lookup(embedding, x_)
        biases = tf.get_variable("biases", [FLAGS.n_classes], tf.float32)
        if FLAGS.ran_emb == 1:
            random_emb = tf.get_variable("ran_emb", [FLAGS.ran_word_num, FLAGS.word_dim], tf.float32)
            ran_x = tf.nn.embedding_lookup(random_emb, x_ran)
            x = tf.concat(2,[x, ran_x])
    dp_emb = tf.get_variable('dp_emb', [DP_NUM, FLAGS.feature_emb_size], tf.float32)
    sdp_emb = tf.get_variable('sdp_emb', [SDP_NUM, FLAGS.feature_emb_size], tf.float32)

    if FLAGS.add_sdp == 1:
            sdp = tf.nn.embedding_lookup(sdp_emb, sdp_)
            x = tf.concat(2, [x, sdp])

    if FLAGS.feature == 1:
                pos_emb = tf.get_variable("pos_emb", [POS_NUM, FLAGS.feature_emb_size], tf.float32)
                ner_emb = tf.get_variable("ner_emb", [NER_NUM, FLAGS.feature_emb_size], tf.float32)
                p = tf.nn.embedding_lookup(pos_emb, pos_)
                n = tf.nn.embedding_lookup(ner_emb, ner_)
                cur = tf.concat(2,[x, p, n])
                x = cur
                if FLAGS.encode_dp_path == 1:
                    #ancestor = encode_ancestors(x_,ancestor_mask,dp_father_index,pos_,pos_emb, ner_,ner_emb, seq_len)
                    father_ind_1 = tf.reshape(dp_father_index,[FLAGS.batch_size*max_len,-1])
                    dp_father_relation_1 = tf.reshape(dp_father_relation,[FLAGS.batch_size*max_len,-1])
                    dp_mask_1 = tf.reshape(dp_mask,[FLAGS.batch_size*max_len,-1])
                    batch_len,max_seq_len,ax_anc_len,hash_indices = anc_index_lookup(x_,father_ind_1)
                    #anc_word,anc_pos,anc_ner = lookup_ancestors(x_,batch_len,pos_,ner_,hash_indices)
                    
                    pos_looktable = tf.reshape(pos_,[-1])
                    word_looktable = tf.reshape(x_,[-1])
                    ner_looktable = tf.reshape(ner_,[-1])
                    anc_word = tf.reshape(tf.gather(word_looktable,hash_indices),[batch_len,-1])
                    anc_pos = tf.reshape(tf.gather(pos_looktable,hash_indices),[batch_len,-1])
                    anc_ner = tf.reshape(tf.gather(ner_looktable,hash_indices),[batch_len,-1])
                    
                    anc_word = tf.nn.embedding_lookup(embedding, anc_word)
                    anc_pos = tf.nn.embedding_lookup(pos_emb, anc_pos)
                    anc_ner = tf.nn.embedding_lookup(ner_emb, anc_ner)#[batch_size*seq_len,max_anc_len,feature_emb_size]
                    #anc_feature = tf.concat(2,[anc_word,anc_pos,anc_ner])

                    dp_father_relation_emb = tf.nn.embedding_lookup(dp_emb,dp_father_relation_1)
                    anc_feature = tf.concat(2,[anc_word,anc_pos,dp_father_relation_emb])
                   # cur = tf.reshape(cur, [FLAGS.batch_size*max_len,1,-1])
                   # anc_input = tf.concat(1,[cur, anc_feature])

                    #anc_seq_len至少为1,包含当前节点,此时为根节点
                    '''
                    with tf.variable_scope('encode_lstm'):
                        #father_index对于padding的word加了一个路径为[1]
                        anc_seq_len = tf.reduce_sum(tf.sign(father_ind_1),axis = 1)
                        encode_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.encode_hidden,
                                            state_is_tuple = True, activation = tf.nn.relu)
                        encode_cell_fw = tf.nn.rnn_cell.DropoutWrapper(encode_cell_fw, output_keep_prob = 0.5, seed = SEED)

                        encode_outputs, _ = tf.nn.bidirectional_dynamic_rnn(encode_cell_fw, \
                                anc_feature, anc_seq_len, dtype = tf.float32)
                    '''
                    encoder = Encoder()
                    encode_v = encoder.lstm_encoder(dp_mask_1,anc_feature)
                    encode_vec = tf.reshape(encode_v, [FLAGS.batch_size,max_seq_len,FLAGS.encode_hidden])
                    x = tf.concat(2,[x,encode_vec])
                if FLAGS.encode_sdp_path == 1:
                    #anc_word,anc_pos,anc_ner,max_seq_len,max_anc_len = anc_index_lookup(x_,sdp_father_index,pos_,ner_)
                    _,max_seq_len,max_anc_len,hash_indices = anc_index_lookup(x_,sdp_father_index,batch_range = batch_range)
                    #anc_word,anc_pos,anc_ner = lookup_ancestors(x_,batch_len,pos_,ner_,hash_indices)
                    batch_len = tf.shape(sdp_father_index)[0]
                    pos_looktable = tf.reshape(pos_,[-1])
                    word_looktable = tf.reshape(x_,[-1])
                    ner_looktable = tf.reshape(ner_,[-1])
                    pack = tf.stack([batch_len,-1])
                    anc_word = tf.reshape(tf.gather(word_looktable,hash_indices),pack)
                    anc_pos = tf.reshape(tf.gather(pos_looktable,hash_indices),pack)
                    anc_ner = tf.reshape(tf.gather(ner_looktable,hash_indices),pack)
                    anc_word = tf.nn.embedding_lookup(embedding, anc_word)
                    anc_pos = tf.nn.embedding_lookup(pos_emb, anc_pos)
                    #[batch_size*seq_len,max_path_num*max_anc_len,emb_size]
                    #anc_ner = tf.nn.embedding_lookup(ner_emb, anc_ner)
                    #[batch_size*seq_len,max_path_num*max_anc_len,emb_size]
                    
                    sdp_father_relation_emb = tf.nn.embedding_lookup(sdp_emb,sdp_father_relation)
                    anc_feature = tf.concat(2,[anc_word,anc_pos,sdp_father_relation_emb])
            
                    encoder = Encoder()
                    encode_v = encoder.lstm_encoder(sdp_father_index,anc_feature,sdp_seq_len)
                    #[path_num,encode_hidden]

                    '''
                    padding MAX_PATH_NUM-REAL_PATH_NUM条path
                    '''
                    encode_v = tf.concat(0,[encode_v,sdp_padding_path])
                    
                    encode_v = tf.gather(encode_v,sdp_path_hash)

                    encode_vec = tf.reshape(encode_v,[FLAGS.batch_size*max_len,MAX_PATH_NUM,-1,1])
                    #[batch_size*max_len,max_path_num,hidden_size,1]
                    pool_input = tf.nn.max_pool(encode_vec,ksize = [1,3,1,1],strides = [1,3,1,1],padding = 'VALID')
                    #shape = tf.stack([FLAGS.batch_size,max_len,-1])
                    sdp_encode_vec = tf.reshape(pool_input,[FLAGS.batch_size,max_len,FLAGS.encode_hidden])
                    #用-1报depth的错
                    x = tf.concat(2,[x,sdp_encode_vec])
                    

    if FLAGS.add_sdp_anc == 1:
        cur_father_emb = tf.nn.embedding_lookup(embedding, cur_father_lookup(x_,cur_sdp_father))
        x = tf.concat(2,[x,cur_father_emb])

    if FLAGS.BiLSTM == 0:
        #with tf.device('/gpu:2'):
        with tf.device('/cpu:0'):
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
    #with tf.device('/gpu:2'):
    with tf.device('/cpu:0'):
        weights = tf.get_variable("weights", weight_shape, tf.float32,
                    initializer = tf.truncated_normal_initializer(stddev=0.01, seed = SEED))
        if FLAGS.add_sdp_in_crf:
            crf_sdp_embedding = tf.get_variable("crf_sdp",[SDP_NUM,FLAGS.feature_emb_size], tf.float32)
            crf_sdp = tf.nn.embedding_lookup(crf_sdp_embedding, sdp_)
            crf_sdp = tf.reshape(crf_sdp,[-1,FLAGS.feature_emb_size])
            outputs = tf.concat(1,[outputs, crf_sdp])
            crf_weights = tf.get_variable("crf_weights", [FLAGS.feature_emb_size, FLAGS.n_classes],\
                tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.01, seed = SEED))
            weights = tf.concat(0,[weights, crf_weights])
        if FLAGS.add_sdp_anc_in_crf:
            cur_father_emb = tf.nn.embedding_lookup(embedding, cur_father_lookup(x_,cur_sdp_father))
            cur_father_emb = tf.reshape(cur_father_emb, [-1, FLAGS.word_dim])
            outputs = tf.concat(1,[outputs, cur_father_emb])
            crf_weights = tf.get_variable("father_anc_weights", [FLAGS.word_dim, FLAGS.n_classes],\
                tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.01, seed = SEED))
            weights = tf.concat(0, [weights, crf_weights])
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
        #with tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) as sess:
            saver = tf.train.Saver()
            best_F = 0
            sess.run(tf.global_variables_initializer())
            logConfig(FLAGS.log_path + file_tail)
            inf = float("-inf")
            for step in range(FLAGS.epoch_step):
                ave_cost = 0
                for i in range(0, train_data.batch_num):
                    _,_,x_pad, ran_x_pad, y_pad, mask_feed, pos_pad, ner_pad, sdp_pad,\
                    path,cur_father,sdp_path,sdp_path_relation,dp_relation,sdp_f_hash_ind,\
                    dp_path_mask,sdp_seq_len_,batch_range_ = get_pad(train_data)
                    get_padding_path_num(sdp_path,sdp_f_hash_ind)
                    padding_path = inf*np.ones([padding_path_num,FLAGS.encode_hidden],dtype = np.float32)
                    #anc_feature_v,dp_mask_1_v = sess.run([dp_mask_1,anc_feature],
                    #encode_v_output_v,encode_vec_v,padding_path_v= sess.run([encode_v_output,encode_vec,sdp_padding_path],\
                    loss,output_np,_ = sess.run([cost,outputs,optimizer],
                            feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,dp_father_index:path,\
                            dp_father_relation:dp_relation,cur_sdp_father:cur_father,sdp_path_hash:sdp_f_hash_ind,\
                            sdp_father_relation:sdp_path_relation,sdp_father_index:sdp_path,\
                            sdp_padding_path:padding_path,\
                            dp_mask:dp_path_mask,sdp_seq_len:sdp_seq_len_,batch_range:batch_range_,\
                        mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1-FLAGS.dropout})
                    ave_cost += loss
                logging.info("step %d, training  loss : %g" % (step, ave_cost/train_data.batch_num))
                num = 0
                cor_num = 0
                ave_cost = 0
                tmp_out = open(FLAGS.output_path + '/dev/'+file_tail+"tmp_out" ,'w')
                for i in range(0, dev_data.batch_num):
                    dev_x,dev_y,x_pad, ran_x_pad, y_pad, mask_feed, pos_pad, ner_pad, sdp_pad,\
                     path,cur_father,sdp_path,sdp_path_relation,dp_relation,sdp_f_hash_ind,\
                     dp_path_mask,sdp_seq_len_,batch_range_  = get_pad(dev_data)
                    get_padding_path_num(sdp_path,sdp_f_hash_ind)
                    padding_path = inf*np.ones([padding_path_num,FLAGS.encode_hidden],dtype = np.float32)
                    if not FLAGS.CRF:
                        for yi in dev_y:
                            num += len(yi)
                        loss,seq_length,ind,correct_num = sess.run([cost,seq_len,indices,correct],\
                                feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,dp_father_index:path,\
                                dp_father_relation:dp_relation,cur_sdp_father:cur_father,sdp_path_hash:sdp_f_hash_ind,\
                                sdp_father_relation:sdp_path_relation,sdp_father_index:sdp_path,\
                                sdp_padding_path:padding_path,\
                                dp_mask:dp_path_mask,sdp_seq_len:sdp_seq_len_,batch_range:batch_range_,\
                            mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1})
                        cor_num += correct_num
                    else:
                        loss,seq_length,correct_num,tran_matrix,score, ind = sess.run([cost,seq_len,correct,\
                            transition_params,unary_score, indices],\
                            feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,dp_father_index:path,\
                            dp_father_relation:dp_relation,cur_sdp_father:cur_father,sdp_path_hash:sdp_f_hash_ind,\
                            sdp_father_relation:sdp_path_relation,sdp_father_index:sdp_path,\
                            sdp_padding_path:padding_path,\
                            dp_mask:dp_path_mask,sdp_seq_len:sdp_seq_len_,batch_range:batch_range_,\
                            mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1})
                        cor_label, label_num, _ = crf_evaluate(seq_length,tran_matrix,score,y_pad)
                        cor_num += cor_label
                        num += label_num
                    ave_cost += loss
                    for k in range(0,FLAGS.batch_size):
                        for j in range(0,seq_length[k]):
                            tmp_out.write(word_dict[dev_x[k][j]] + '\t' + label_dict[dev_y[k][j]] + '\t' + label_dict[ind[k*max_len+j][0]] + '\n')
                        tmp_out.write('\n')
                    tmp_out.flush()
                tmp_out.close()
                best_F,dev_F, accuracy, savemodel = evaluate(best_F,file_tail,'dev')
                if savemodel:
                    saver.save(sess, FLAGS.saver_path + file_tail)
                    os.system("cp " + FLAGS.output_path + '/dev/'+file_tail + "tmp_out " + FLAGS.output_path +'/dev/'+ file_tail)
                logging.info("step %d , dev_loss: %g,accuracy: %s, dev_F: %g" % (step,ave_cost/dev_data.batch_num,accuracy,dev_F))
    else:
        with tf.Session(config = gpu_config()) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, FLAGS.saver_path + file_tail)
            #label_dict = pkl.load(open(FLAGS.label_dict_path,'r'))
            #word_dict = pkl.load(open(FLAGS.word_dict_path,'r'))
            out = open(FLAGS.output_path + '/test/'+file_tail+'tmp_out' ,'w')
            num = 0
            cor_num = 0
            ave_cost = 0
            best_F = 0
            inf = float("-inf")
            for i in range(0, test_data.batch_num):
                test_x,test_y,x_pad, ran_x_pad, y_pad, mask_feed, pos_pad, ner_pad, sdp_pad,\
                 path,cur_father,sdp_path,sdp_path_relation,dp_relation,sdp_f_hash_ind,\
                 dp_path_mask,sdp_seq_len_,batch_range_  = get_pad(test_data)
                get_padding_path_num(sdp_path,sdp_f_hash_ind)
                padding_path = inf*np.ones([padding_path_num,FLAGS.encode_hidden],dtype = np.float32)
                if not FLAGS.CRF:
                    for yi in test_y:
                        num += len(yi)
                    loss,seq_length,ind,correct_num = sess.run([cost,seq_len,indices,correct],\
                            feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,dp_father_index:path,\
                            dp_father_relation:dp_relation,cur_sdp_father:cur_father,sdp_path_hash:sdp_f_hash_ind,\
                            sdp_father_relation:sdp_path_relation,sdp_father_index:sdp_path,\
                            sdp_padding_path:padding_path,\
                            dp_mask:dp_path_mask,sdp_seq_len:sdp_seq_len_,batch_range:batch_range_,\
                        mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1})
                    cor_num += correct_num
                else:
                    loss,seq_length,correct_num,tran_matrix,score = sess.run([cost,seq_len,correct,\
                        transition_params,unary_score],\
                        feed_dict = {x_:x_pad, x_ran:ran_x_pad,y_:y_pad,pos_:pos_pad,dp_father_index:path,\
                        dp_father_relation:dp_relation,cur_sdp_father:cur_father,\
                        sdp_father_relation:sdp_path_relation,sdp_father_index:sdp_path,sdp_path_hash:sdp_f_hash_ind,\
                        dp_mask:dp_path_mask,sdp_seq_len:sdp_seq_len_,batch_range:batch_range_,\
                        sdp_padding_path:padding_path,\
                        mask:mask_feed,ner_:ner_pad,sdp_:sdp_pad,out_keep_prob:1})
                    cor_label_num, label_num, ind = crf_evaluate(seq_length,tran_matrix,score,y_pad)
                    cor_num += cor_label_num
                    num += label_num
                ave_cost += loss
                for k in range(0,FLAGS.batch_size):
                    for j in range(0,seq_length[k]):
                        out.write(word_dict[test_x[k][j]] + '\t' + label_dict[test_y[k][j]] + '\t' + label_dict[ind[k*max_len+j][0]] + '\n')
                    if seq_length[k] != 0:
                        out.write('\n')
                out.flush()
            best_F,test_F,accuracy, savemodel = evaluate(best_F, file_tail,'test')
            if savemodel:
                os.system("cp " + FLAGS.output_path +'/test/'+ file_tail + "tmp_out " + FLAGS.output_path +'/test/'+ file_tail)
            print "test_loss: %g,accuracy: %s, test_F: %g" % (ave_cost/test_data.batch_num,accuracy,test_F)
            print file_tail
            out.close()


def main(_):
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    dynamic_rnn()

if __name__ == '__main__':
    tf.app.run()
