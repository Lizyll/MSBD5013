import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

def sigmoid(z):
	return (1 / (1 + np.exp(-z)))

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    
    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0], data[pos][:,2], marker='+', c='k', s=60, linewidth=2, label=label_pos, norm=normalize)
    axes.scatter(data[neg][:,0], data[neg][:,2], c='b', s=60, label=label_neg, norm=normalize)

    #axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos, norm=normalize)
    #axes.scatter(data[neg][:,0], data[neg][:,1], c='b', s=60, label=label_neg, norm=normalize)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);
    plt.show()


# load data
data = pd.read_csv('test_data2.csv', header = 0)


# logistic regression
# divide into 9 distance groups
dis_1000 = data[data['distance'] == 1000][['besttime', 'horseweight', 'ind_win']].to_numpy()
dis_1200 = data[data['distance'] == 1200][['finishm', 'hid', 'ind_win']].to_numpy()
dis_1400 = data[data['distance'] == 1400][['finishm', 'horseweight', 'ind_win']].to_numpy()
dis_1600 = data[data['distance'] == 1600][['finishm', 'horseweight', 'ind_win']].to_numpy()
dis_1650 = data[data['distance'] == 1650][['finishm', 'horseweight', 'ind_win']].to_numpy()
dis_1800 = data[data['distance'] == 1800][['finishm', 'hid', 'ind_win']].to_numpy()
dis_2000 = data[data['distance'] == 2000][['finishm', 'horseweight', 'ind_win']].to_numpy()
dis_2200 = data[data['distance'] == 2200][['finishm', 'horseweight', 'ind_win']].to_numpy()
dis_2400 = data[data['distance'] == 2400][['finishm', 'horseweight', 'ind_win']].to_numpy()
#print(dis_1000)


# 1000 logistic

# replace NaN with avg of column
colmean1 = np.nanmean(dis_1000, axis=0)
print(colmean1)
inds = np.where(np.isnan(dis_1000))
dis_1000[inds] = np.take(colmean1, inds[1])
#print(dis_1000)

colmean2 = np.nanmean(dis_1200, axis=0)
inds = np.where(np.isnan(dis_1200))
dis_1200[inds] = np.take(colmean2, inds[1])

colmean3 = np.nanmean(dis_1400, axis=0)
inds = np.where(np.isnan(dis_1400))
dis_1400[inds] = np.take(colmean3, inds[1])

colmean6 = np.nanmean(dis_1800, axis=0)
inds = np.where(np.isnan(dis_1800))
dis_1800[inds] = np.take(colmean6, inds[1])

colmean8 = np.nanmean(dis_2200, axis=0)
inds = np.where(np.isnan(dis_2200))
dis_2200[inds] = np.take(colmean8, inds[1])

X1000 = np.c_[np.ones((dis_1000.shape[0], 1)), dis_1000[:, 0:2]]
y1000 = np.c_[dis_1000[:,2]]
normalize = matplotlib.colors.Normalize(vmin=-1, vmax=1)
plotData(dis_1000, 'finish time', 'hid', 'win', 'not win')

X1200 = np.c_[np.ones((dis_1200.shape[0], 1)), dis_1200[:, 0:2]]
y1200 = np.c_[dis_1200[:,2]]
normalize = matplotlib.colors.Normalize(vmin=-1, vmax=1)

#plotData(dis_1200, 'finish time', 'hid', 'win', 'not win')

#plotData(dis_1400, 'finish time', 'hid', 'win', 'not win')

#plotData(dis_2200, 'finish time', 'hid', 'win', 'not win')
