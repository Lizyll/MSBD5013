import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures



def datediff(i):
	if i <= 33.7:
		return 1
	else:
		return 0

def horseid(i):
	if i <= 0:
		return 'Invalid horse id input'
	elif i <= 6:
		return 1
	else:
		return 0
data = pd.read_csv('test_data2.csv', header = 0)
#print(data)

'''
def computeCost(X, y, theta=[[0],[0]]):
	m = y.size
	J = 0
	h = X.dot(theta)
	J = 1/(2*m) * np.sum(np.square(h-y))
	return J
'''
# total match of jockey and trainer
jockeydict = data.jname.value_counts().to_dict()
trainerdict = data.tname.value_counts().to_dict()


# winning dict of jockey and trainer
df = data[data['ind_win'] == 1] # 1st horse pool
df2 = data[data['ind_pla'] == 1]  # first 3 horse pool
'''
print(df['datediff'].mean())
print(df2['datediff'].mean())
print(df['age'].mean())
print(df2['age'].mean())
'''
jockeywindict = df.jname.value_counts().to_dict()
trainerwindict = df.tname.value_counts().to_dict()

# jockey win_odds
jocwin = dict()
for key, val in jockeydict.items():
	if key in jockeywindict:
		value = jockeywindict.get(key) / val
		jocwin.update({key: value})
#print('Jockey winning odds: ', jocwin)
#print('len of jocwin', len(jocwin))

# trainer win_odds
trainwin = dict()
for key, val in trainerdict.items():
	if key in trainerwindict:
		value = trainerwindict.get(key) / val
		trainwin.update({key: value})
#print('Trainer winning odds: ', trainwin)

# 40 percentile of jockey
joc = [ v for k, v in jocwin.items()] # convert to a list 
joc40 = np.percentile(joc, 40, axis=0)
#print(joc40)

# jockey the first 40% winning odds mark as 1, else: 0
jocwin40 = dict()
for k, v in jocwin.items():
	if v >= joc40:
		jocwin40.update({k: 1})
	else:
		jocwin40.update({k : 0})
print(jocwin40)

# 40 percentile of trainer
tra = [ v for k, v in trainwin.items()]
tra40 = np.percentile(tra, 40, axis=0)
#print(tra40)
# trainer the first 40% winning odds mark as 1, else: 0
trainwin40 = dict()
for k, v in trainwin.items():
	if v >= tra40:
		trainwin40.update({k: 1})
	else:
		trainwin40.update({k: 0})
#print(trainwin40)


# track features
tmp = data['track'].unique()
trackdict = dict()
for i in tmp:
	if i == 'TURF':
		trackdict.update({i: 1})
	else:
		trackdict.update({i: 0})
#print(trackdict)


# X: besttime, y: finishm -> linear regression

# set X ,y
#X1 = np.c_[np.ones(len(data)), data['besttime'].to_numpy()]
X = data['besttime'].to_numpy().T
X = np.nan_to_num(X)
#print(np.where(np.isnan(X)))
#print(X.shape)
y = data['finishm'].to_numpy().T
y = np.nan_to_num(y)
print(X.shape)
#print(y.shape)
plt.scatter(X, y, c='b', marker='x', linewidths=1)
plt.xlabel('Best time(s)')
plt.ylabel('Finish time(s)')
#plt.show()


ran = np.arange(55, 153)
regr = LinearRegression(fit_intercept=False).fit(X.reshape(-1,1), y)
print('Coefficients: ', regr.coef_)
#print('Mean squred error: %.2f' % mean_squared_error(X, y))
plt.plot(ran, regr.intercept_+regr.coef_*ran, label='Linear Regression')
plt.show()

print(jocwin40.get('M F Poon'))
