#coding:utf-8

import cPickle as pkl

def calculate(data_type = 'sdp',file_type = 'train'):
    str_tail = "_all_path_to_root"
    file_name = file_type + '_' + data_type + str_tail
    paths = pkl.load(open(file_name,'r'))
    count = dict()
    for sen in paths:
        for w in sen:
            path_num = len(w)
            if count.has_key(path_num):
                count[path_num]+=1
            else:
                count[path_num] = 1
    print file_type,file_type,"max:%d"%(max(count))
    print count

if __name__ == "__main__":
    calculate()
    calculate(file_type = 'test')
    calculate(file_type = 'dev')
