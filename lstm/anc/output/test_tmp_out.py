#coding:utf-8
import commands

def test():
	datas = open('tmp_out','r').read().strip().split('\n\n')
	for data in datas:
		lines = data.split('\n')
		for line in lines:
			if len(line.split('\t')) != 3:
				print data

def test_perl():
	cmd = "perl ../../conlleval.pl -d \"\\t\" < " + "./tmp_out"
	status,output = commands.getstatusoutput(cmd)
	print status,output

if __name__ == '__main__':
	test_perl()
	test()