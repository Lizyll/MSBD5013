import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
from sklearn import preprocessing
from operator import itemgetter
import operator
from scipy.spatial import distance


def rmse(predictions, targets):
	return np.sqrt(np.mean((predictions-targets)**2))

# 3 layers
class NeuralNetWork(object):
	# input layer
	def __init__(self):

		# read in
		df = pd.read_csv('Preprocessing/sheet2.csv')
		y_pla = df['ind_pla']
		y_win = df['ind_win']

		self.fristpara = 10
		self.secondpara = 8

		# 10 parameters
		X = np.dstack((df['age'], df['tname'], df['jname'], df['rating'], 
						df['track'], df['bardraw'], df['besttime'], df['going'], 
						df['win_t5'], df['place_t5']))
		X = np.reshape(X, (-1, self.fristpara))
		#print(X)
		Standard_Scaler = preprocessing.StandardScaler()
		X = Standard_Scaler.fit_transform(X)
		dt = pd.DataFrame(X)

		# def weights and biases
		self.weight1 = []  #hidden layer weights
		self.bias1 = []
		self.weight2 = []  #output layer weights
		self.bias2 = []
		self.Xdata = []
		
		
		self.Xdata = dt.values
		self.y_pladata = y_pla.values             ##############pla win prediction
		self.y_windata = y_win.values

		self.xs = tf.placeholder(tf.float32, [1, None])
		self.ys = tf.placeholder(tf.float32, [1, None])
		self.xws = tf.placeholder(tf.float32, [1, None])
		self.yws = tf.placeholder(tf.float32, [1, None])

		self.weight1 = tf.Variable(tf.random_normal([self.fristpara, self.secondpara]))
		self.bias1 = tf.Variable(tf.zeros([1, self.secondpara]) + 0.1)
		self.weight2 = tf.Variable(tf.random_normal([self.secondpara, 1]))
		self.bias2 = tf.Variable(tf.zeros([1, 1]) + 0.1)

		self.pdata = []

	# hidden layer
	def layer2(self, inputs, activation_function=tf.nn.relu):
		z = tf.matmul(inputs, self.weight1) + self.bias1
		if activation_function is None:
			outputs = z
		else:
			outputs = activation_function(z)
		return outputs

	# output layer
	def layer3(self, inputs, activation_function=tf.nn.sigmoid):
		z = tf.matmul(inputs, self.weight2) + self.bias2
		if activation_function is None:
			outputs = z
		else:
			outputs = activation_function(z)
		return outputs

	# train
	def trainwin(self):
	
		y2 = self.layer2(self.xws)
		self.prediction1 = self.layer3(y2)
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.yws - self.prediction1), reduction_indices=[1]))
		train_step = tf.train.RMSPropOptimizer(0.1).minimize(loss)
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

		# train win
		for step in range(1):
			for i in range(len(self.Xdata)):
				x_dtwin = self.Xdata[i].reshape(1, -1)
				y_dtwin = self.y_windata[i].reshape(1, -1)
				self.sess.run(train_step, feed_dict={self.xws: x_dtwin, self.yws: y_dtwin})
				self.sess.run(loss, feed_dict={self.xws: x_dtwin, self.yws: y_dtwin})
			if step % 50 == 0:
				self.sess.run(self.prediction1, feed_dict={self.xws: x_dtwin, self.yws: y_dtwin})
				self.sess.run(loss, feed_dict={self.xws: x_dtwin, self.yws:y_dtwin})

	def trainpla(self):

		y1 = self.layer2(self.xs)
		self.prediction = self.layer3(y1)
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.prediction), reduction_indices=[1]))
		train_step = tf.train.RMSPropOptimizer(0.1).minimize(loss)
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
		
		# train pla
		for step in range(1):
			for i in range(len(self.Xdata)):
				x_dt = self.Xdata[i].reshape(1, -1)
				y_dt = self.y_pladata[i].reshape(1, -1)
				self.sess.run(train_step, feed_dict={self.xs: x_dt, self.ys: y_dt})
				self.sess.run(loss, feed_dict={self.xs: x_dt, self.ys: y_dt})
			if step % 50 == 0:
				self.sess.run(self.prediction, feed_dict={self.xs: x_dt, self.ys: y_dt})
				self.sess.run(loss, feed_dict={self.xs: x_dt, self.ys:y_dt})
		
	
	def test(self):
		print('====================testing part====================')

		tdata = pd.read_csv('Preprocessing/sample.csv')
		tX = np.dstack((tdata['age'], tdata['tname'], tdata['jname'], tdata['rating'], 
						tdata['track'], tdata['bardraw'], tdata['besttime'], tdata['going'], 
						tdata['win_t5'], tdata['place_t5']))
		tX = np.reshape(tX, (-1, self.fristpara))
		Standard_Scaler2 = preprocessing.StandardScaler()
		tX = Standard_Scaler2.fit_transform(tX)

		df_tX = pd.DataFrame(tX)
		self.td = df_tX.values

		plapro = []
		winpro = []
		pla_outcome = []
		for i in range(len(self.td)):
			px = self.td[i].reshape(1, -1)
			pypla = self.sess.run(self.prediction, feed_dict={self.xs: px})			
			plapro.append(pypla)
			

		
		for i in range(len(self.td)):
			px1 = self.td[i].reshape(1, -1)
			pywin = self.sess.run(self.prediction1, feed_dict={self.xws: px1})
			winpro.append(pywin)
		#print(winpro)
		

		tmp = []
		tmp1 = []
		for i in plapro:
			for j in i:
				for z in j:
					tmp.append(z)
					if z >= 0.003:
						pla_outcome.append(1)
					else:
						pla_outcome.append(0)
		for i in winpro:
			for j in i:
				for z in j:
					tmp1.append(z)
		tdata['winprob'] = tmp1
		tdata['plaprob'] = tmp

		tdata['winstake'] = np.zeros(len(tmp), dtype=int)
		tdata['plastake'] = np.zeros(len(tmp), dtype=int)
		

		tdata.to_csv('testing.csv', index=False, encoding='utf-8')
		d = pd.read_csv('testing.csv', usecols=['ind_win', 'winprob', 'ind_pla', 'plaprob'])

		a1 = d['ind_win'].to_numpy()
		a2 = d['winprob'].to_numpy()
		a3 = d['ind_pla'].to_numpy()
		a4 = d['plaprob'].to_numpy()

		RMSEwin = rmse(a2, a1)
		RMSEpla = rmse(a1, a3)
		print('RMSEwin: ', RMSEwin, 'RMSEpla: ', RMSEpla)
		print(len(tdata))
		
if __name__ == '__main__':
	A = NeuralNetWork()
	A.trainwin()

	A.trainpla()
	A.test()

