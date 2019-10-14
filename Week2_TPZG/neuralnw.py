# We chose 11 features and applied neural network to get the weights 
# of each feature
# There are one hidden layer and one output layer
# The testing part is within the NeuralNetWork Class


import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
from sklearn import preprocessing
from operator import itemgetter
import operator


# 3 layers
class NeuralNetWork(object):
	# input layer
	def __init__(self):

		# read in
		df = pd.read_csv('Preprocessing/data_class5.csv')
		y_pla = df['ind_pla']
		y_win = df['ind_win']

		self.fristpara = 11
		self.secondpara = 7

		# 10 parameters
		X = np.dstack((df['hid'], df['age'], df['horseweight'], df['exweight'], 
						df['horseweightchg'], df['besttime'], df['distance'],
						df['jname'], df['tname'], df['rating'], df['ratechg']))
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
		#self.y_windata = y.win.values

		self.xs = tf.placeholder(tf.float32, [1, None])
		self.ys = tf.placeholder(tf.float32, [1, None])
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
	def train(self):
		y1 = self.layer2(self.xs)
		self.prediction = self.layer3(y1)
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.prediction), reduction_indices=[1]))
		train_step = tf.train.RMSPropOptimizer(0.1).minimize(loss)
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

		for step in range(10):
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

		predicted_y = []
		for i in range(len(self.Xdata)):
			x_dt = self.Xdata[i].reshape(1, -1)
			pre_y = self.sess.run(self.prediction, feed_dict={self.xs: x_dt})
			
			if pre_y >= 0.4:
				pre_y = 1
			else:
				pre_y = 0
			
			predicted_y.append(pre_y)

		ob_y = self.y_pladata
		#print('observed y: ', self.y_pladata)
		#print(ob_y)
		#print(predicted_y)

		error = np.sum(np.abs(ob_y - predicted_y))
		err_rate = error / len(ob_y)
		print('total number: ', len(ob_y), 'right number: ', len(ob_y) - error, 'right rate: ', 1 - err_rate)


		observedwin = 0
		predictwin = 0

		for i in range(len(self.y_pladata)):
			if self.y_pladata[i] == 1:
				observedwin += 1
				if predicted_y[i] == 1:
					predictwin += 1

		print(observedwin, predictwin)

		for i in range(len(self.Xdata)):
			x_dt = self.Xdata[i].reshape(1, -1)
			self.sess.run(self.prediction, feed_dict={self.xs: x_dt})


	def predict(self):
		# Predicted result
		predictiondata = pd.read_csv('Oct16data/match1.csv').fillna(0)
		input_X = np.dstack((predictiondata['hid'], predictiondata['age'], predictiondata['horseweight'],
							predictiondata['exweight'], predictiondata['horseweightchg'], predictiondata['besttime'], 
							predictiondata['distance'], predictiondata['jname'], predictiondata['tname'], 
							predictiondata['rating'], predictiondata['ratechg']))
		input_X = np.reshape(input_X, (-1, self.fristpara))
		Standard_Scaler2 = preprocessing.StandardScaler()
		input_X = Standard_Scaler2.fit_transform(input_X)

		df_input_X = pd.DataFrame(input_X)

		self.pdata = df_input_X.values

		print('====================prediction part====================')


		outcome_y = []
		for i in range(len(self.pdata)):
			px = self.pdata[i].reshape(1, -1)
			py = self.sess.run(self.prediction, feed_dict={self.xs: px})
			'''
			if py >= 0.4:
				py = 1
			else:
				py = 0
			'''
			outcome_y.append(py)

		predictiondata['ind_pla'] = pd.Series(outcome_y)

		#print(predictiondata['ind_pla'])
		# sort()
		
		tmp = predictiondata['ind_pla'].to_list()
		tmp1 = list()
		tmp2 = dict()
		for i in tmp:
			for j in i:
				for z in j:
					tmp1.append(z)
		
		leng = len(tmp1)
		
		for i in range(0, 10):
			tmp2.update({str(i+1): tmp1[i]})
		tmp3 = sorted(tmp2.items(), key=operator.itemgetter(1), reverse=True)
		#print(tmp3[:3])
		tmp4 = [x[0] for x in tmp3[:3]]
		tmp5 = [x[1] for x in tmp3[:3]]
		tmp6 = []
		tmp7 = []
		#print(tmp4)
		for i in range(1, leng+1):
			j = str(i)
			if j in tmp4:
				tmp6.append(1)
			else:
				tmp6.append(0)
		for i in range(1, leng+1):
			j = str(i)
			if j in tmp4[0]:
				tmp7.append(1)
			else:
				tmp7.append(0)
		
		#predictiondata['plapro'] = tmp1
		predictiondata['ind_pla'] = tmp6
		predictiondata['ind_win'] = tmp7
		#predictiondata['ind_val'] = tmp1

		predictiondata.to_csv('Oct16data/match1.csv', index=False, encoding='utf-8')
			
		print('Prediction.csv has been gernerated.')	






		
		

if __name__ == '__main__':
	A = NeuralNetWork()
	A.train()
	#A.test()
	A.predict()

