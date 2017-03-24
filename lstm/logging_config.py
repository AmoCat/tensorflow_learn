#!usr/bin/env python
#coding:utf-8

import logging

def logConfig(file_name = './log/log'):
    logging.basicConfig(
            level = logging.DEBUG,
            format   = '%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
            datefmt = '%Y-%m-%d %A %H:%M:%S',
            filename = file_name,
            filemode = 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')
    console.setFormatter(formatter)

    logging.getLogger().addHandler(console)


