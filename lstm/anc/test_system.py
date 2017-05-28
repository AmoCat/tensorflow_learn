#coding:utf-8

import os
import commands
import re

if __name__ == "__main__":
    cmd = "perl ../conlleval.pl -d \"\\t\" < ./output/baseline-L2-0.0001-h128-fea-1-50-epoch-40-b16"
    #res = os.system(cmd)
    (status, output) = commands.getstatusoutput(cmd)
    l = output.split('\n')
    res = l[1]
    F = l[1].split(' ')[-1]
    print res,F
    #file = open("./PRFresult/test",'w')
    #file.write(output)
    print status,output
