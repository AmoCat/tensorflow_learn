#coding:utf-8
from copy import deepcopy
import numpy
import tensorflow as tf

MAX_HEAD_NUM = 3

def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    return config

#one father node
def padding_path(data, max_seg = 54,padding_num = 1):
    max_path_len = 0
    for sen in data:
        path_len = max([len(path) for path in sen])
        max_path_len = path_len if path_len > max_path_len else max_path_len
    batch_size = len(data)
    if padding_num == 1:
        path_pad = numpy.ones([batch_size,max_seg,max_path_len], numpy.int32)
    else:
        path_pad = numpy.zeros([batch_size,max_seg,max_path_len], numpy.int32)
    actual_head_num = []
    for i in range(0, len(data)):
        sen = data[i]
        for j in range(0, max_seg):
            if j >= len(sen):#对句子padding word
                path_pad[i][j][0] = 1
                continue
            path = sen[j]
            #cur_path_len = np.sign(sen).sum()
            cur_path_len = len(path)#非实际长度，可能有[[0]]的path
            batch_path = deepcopy(data[i][j])
            batch_pad = [padding_num for _ in range(max_path_len - cur_path_len)]
            if cur_path_len < max_path_len:
                batch_path.extend(batch_pad)
            path_pad[i][j] = batch_path
    path_pad = numpy.reshape(path_pad, [batch_size,-1])
    return path_pad

'''
pad sdp index and relations,index补1,relation补0
def padding_sdp_path(data, max_path_num = 3, max_seg = 54, padding_num = 1):
    max_path_len = 0
    batch_size = len(data)
    for sen in data:
        for w in sen:
            for path in w:
                if len(path) > max_path_len:
                    max_path_len = len(path)
    if padding_num == 0:
        path_pad = numpy.zeros([batch_size,max_seg,max_path_num,max_path_len])
    else:
        path_pad = numpy.ones([batch_size,max_seg,max_path_num,max_path_len])
    for i in range(batch_size):
        sen = data[i]
        for j in range(max_seg):
            if j >= len(sen):
                continue
            w = data[i][j]
            for k in range(len(w)):
                path = deepcopy(w[k])
                cur_len = len(path)
                pad = [padding_num for _ in range(max_path_len -  cur_len)]
                path.extend(pad)
                path_pad[i][j][k] = path
    path_pad = numpy.reshape(path_pad,[batch_size*max_seg,max_path_num,-1])#[batch_size*max_seg,max_path_num*max_anc_len]
    return path_pad
'''


'''
batch_range 是每个word的每个head节点对应batch_size*seq_len中的index
head不可以类似dp path不传batch_range，因为那种的必须padding到一样的path_leb
即这个需要padding到一样的head_len，但事先不能pad，必须在tf中进行pad,因此
只需要传如hash_indices，并且将mulyihead维度调整为[总head数目]，则需要pad的head个数是
batch_size*seq_len*max_head_num - 总head数目
不需要pad，只需要在少的词部分给每个词补一个父节点
'''
def padding_multihead(data, max_seg = 54, padding_num =1, max_head_num = 3):
    multihead = []
    actual_head_num = []
    batch_range = []
    batch_size = len(data)
    for i in range(len(data)):
        sen = data[i]
        for j in range(max_seg):#对每个词
            if j >= len(sen):
                multihead.append(padding_num)
                actual_head_num.append(1)
                batch_range.append(i*max_seg)
                continue
            w = data[i][j]
            actual_head_num.append(len(w))
            multihead.extend(w)
            batch_range.extend([i*max_seg for _ in range(len(w))])
    
    real_head_num = sum(actual_head_num)
    padding_head_num = batch_size*max_seg*max_head_num - real_head_num
    padding_head_ind = [i for i in range(real_head_num,batch_size*max_seg*max_head_num)]
    head_hash_ind = []
    pre_sum = 0
    start = 0
    for i in range(len(actual_head_num)):
        num = actual_head_num[i]
        batch_id = i/max_seg
        word_id = i/(batch_size*max_seg)
        for j in range(num):
            head_hash_ind.append(pre_sum)
            pre_sum += 1
        while len(head_hash_ind)%(max_head_num) != 0:
            head_hash_ind.append(padding_head_ind[start])
            start += 1

    return multihead,batch_range,head_hash_ind   


            
'''
返回[总路径长度,最大path长度]的path 或relation 以及path_hash_ind
anc_seq_len:[总路径长度]，每条路径的seq_len
batch_range 是每条路径上的每个节点对应的batch_size*seq_len中的index，代表第几个词
'''
def padding_sdp_path(data, max_path_num = 3, max_seg = 54, padding_num = 1):
    max_path_len = 0
    batch_size = len(data)
    for sen in data:
        for w in sen:
            for path in w:
                if len(path) > max_path_len:
                    max_path_len = len(path)
    actual_path_num = []
    paths = []
    anc_seq_len = []
    batch_range = []
    for i in range(batch_size):
        sen = data[i]#每个句子
        word_num = len(sen)
        for j in range(max_seg):
            if j >= len(sen):
                #给缺失的word补一条路径
                paths.append([padding_num for _ in range(max_path_len)])
                actual_path_num.append(1)
                anc_seq_len.append(1)#至少为1否则下边-1后为负数
                #由于path padding了，所以batch_range增加max_path_len个
                batch_range.extend([i*max_seg for _ in range(max_path_len)])
                continue
            w = data[i][j]#每个词
            actual_path_num.append(len(w))
            for k in range(len(w)):#每个词的每条路径
                path = deepcopy(w[k])
                cur_len = len(path)
                anc_seq_len.append(cur_len)
                pad = [padding_num for _ in range(max_path_len -  cur_len)]
                #每条路径上的每个词都有max_path_len个range
                batch_range.extend([i*max_seg for it in range(max_path_len)])
                path.extend(pad)
                paths.append(path)
    #paths:[所有batch_size*max_seq_len个词总路径条数,max_path_len]
    #actual_path_num([batch_size*max_seq_len],每个词的路径条数(补的词为1))
    #print "batch_size:%d,max_seg:%d"%(batch_size,max_seg),"acutal:",actual_path_num
    path_hash_ind = []
    '''
    记录尾部补path后的batch_size*max_seq_len*max_path_num个path如何映射,
    使之可以用reshape变换
    '''
    real_path_num = sum(actual_path_num)
    padding_path_num = batch_size*max_seg*max_path_num - real_path_num
    padding_path_ind = [i for i in range(real_path_num,batch_size*max_seg*max_path_num)]
    start = 0
    pre_sum = 0
    #print "padding_path_num",padding_path_num,"padding_path_ind",padding_path_ind
    for i in range(len(actual_path_num)):
        num = actual_path_num[i]
        batch_id = i/max_seg
        word_id = i/(batch_size*max_seg)
        for j in range(num):
            path_hash_ind.append(pre_sum)
            pre_sum += 1
        while len(path_hash_ind)%(max_path_num) != 0:
            path_hash_ind.append(padding_path_ind[start])
            start += 1
    '''
    print "batch_size:%d,max_seg:%d,max_path_len:%d,max_path_num:%d"%(batch_size,max_seg,max_path_len,max_path_num)
    print "data",data,"path_hash_ind",path_hash_ind
    '''
    batch_range = numpy.array(batch_range)
    anc_seq_len = numpy.array(anc_seq_len)
    batch_range = batch_range
    return paths,numpy.array(path_hash_ind),anc_seq_len,batch_range

def padding_fea(data, max_seg = 54, padding_num = 0):
    batch_size = data.shape[0]
    batch_x = numpy.zeros([batch_size, max_seg],numpy.int32)

    for i in range(batch_size):
        cur_len = len(data[i])
        zeros = [padding_num for _ in range(max_seg - cur_len)]
        batch_x_i = deepcopy(data[i])

        if cur_len < max_seg:
            batch_x_i.extend(zeros)
        batch_x[i] = batch_x_i
    return batch_x

def padding(data,labels,pos,ner,max_seg = 54):
    assert isinstance(data,numpy.ndarray),'the data type should be numpy.ndarray whose dtype is list'
    assert isinstance(labels,numpy.ndarray),'the label type should be numpy.ndarray whose dtype is list'
    assert data.shape == labels.shape,'data shape and labels shape should be same'

    #max_seg = max([len(instance) for instance in data])
    batch_size = data.shape[0]

    batch_x = numpy.zeros([batch_size, max_seg],numpy.int32)
    batch_y = numpy.zeros([batch_size, max_seg],numpy.int32)
    batch_pos = numpy.zeros([batch_size, max_seg],numpy.int32)
    batch_ner = numpy.zeros([batch_size, max_seg],numpy.int32)

    mask = numpy.zeros([batch_size, max_seg],numpy.int32)

    for i in range(batch_size):
        cur_len = len(data[i])
        real_len = numpy.sign(data[i]).sum()
        batch_zeros = [0 for _ in range(max_seg - cur_len)]
        mask_zeros = [0 for _ in range(max_seg - real_len)]
        ones = [1 for _ in range(real_len)]

        batch_x_i = deepcopy(data[i])
        batch_y_i = deepcopy(labels[i])
        batch_pos_i = deepcopy(pos[i])
        batch_ner_i = deepcopy(ner[i])
        if cur_len < max_seg:
            batch_x_i.extend(batch_zeros)
            batch_y_i.extend(batch_zeros)
            batch_pos_i.extend(batch_zeros)
            batch_ner_i.extend(batch_zeros)
        if real_len < max_seg:
            ones.extend(mask_zeros)

        batch_x[i] = batch_x_i
        batch_y[i] = batch_y_i
        batch_pos[i] = batch_pos_i
        batch_ner[i] = batch_ner_i
        mask[i] = ones
    batch_y = batch_y.reshape([batch_size * max_seg])
    mask = mask.reshape([batch_size * max_seg])
    return batch_x,batch_y,batch_pos,batch_ner,mask

if __name__ == '__main__':
    '''
    两个句子，每个句子两个词,第一个句子路径个数1，2，第二个句子路径个数2，1
    '''
    sen_1 = [[[1,2],[1]], [[1]]]
    sen_2 = [[[1]] ]
    data = []
    data.append(sen_1)
    data.append(sen_2)
    padding_multihead(numpy.array(sen_1),max_seg = 2,max_head_num = 3)