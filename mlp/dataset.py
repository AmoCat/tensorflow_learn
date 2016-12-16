#coding:utf-8
import csv
import numpy as np


class Dataset(object):
	def __init__(self, is_train, test_name = "TestSamples1.csv"):
		if is_train:
			tr_np,tr_label_np,te_np,te_label_np = self.read_dataset()
			self.tr_np = tr_np
			self.tr_label_np = tr_label_np
			self.te_np = te_np
			self.te_label_np = te_label_np
		else:
			self.te_np = self.read_testdata(test_name)

		self._index_in_epoch = 0
		self._epoch_completed = 0
		self.num_examples = 20000

	def read_testdata(self,test_name):
		test_s_r = csv.reader(open(test_name, 'r'))
		test_list = [line for line in test_s_r]
		return np.array(test_list, np.float32)

	def read_dataset(self, test_name = "TestSamples1.csv", test_label_name = "TestLabels1.csv"):
		train_s_r = csv.reader(open("TrainSamples.csv", 'r'))
		train_l_r = csv.reader(open("TrainLabels.csv", 'r'))
		test_s_r = csv.reader(open(test_name, 'r'))
		test_l_r = csv.reader(open(test_label_name, 'r'))

		train_list = [line for line in train_s_r]
		train_label_list = [l for l in train_l_r]
		test_list = [line for line in test_s_r]
		test_label_list = [l for l in test_l_r]

		self.tr_np = np.array(train_list, np.float32)
		self.tr_label_np = np.array(train_label_list, np.int32).reshape([len(train_label_list),])
		self.te_np = np.array(test_list, np.float32)
		self.te_label_np = np.array(test_label_list, np.int32).reshape([len(test_label_list),])


		return self.tr_np, self.tr_label_np, self.te_np, self.te_label_np

	def next_batch(self, batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self.num_examples:
			self._epoch_completed += 1
			perm = np.arange(self.num_examples)
			np.random.shuffle(perm)
			self.tr_np = self.tr_np[perm]
			self.tr_label_np = self.tr_label_np[perm]
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self.num_examples
		end = self._index_in_epoch
		#if self._index_in_epoch == 0:
		#	perm = np.arange(self.num_examples)
		#	np.random.shuffle(perm)
		#	self.tr_np = self.tr_np[perm]
		#	self.tr_label_np = self.tr_label_np[perm]
		#self._index_in_epoch += batch_size
		#if self._index_in_epoch >= self.num_examples:
		#	self._index_in_epoch = 0
		#	self._epoch_completed += 1
		#	end = self.num_examples-1
		#else:
		#	end = self._index_in_epoch

		return self.tr_np[start:end],self.tr_label_np[start:end]

if __name__ == "__main__":
	data = Dataset()
