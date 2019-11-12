import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import seaborn as sns
import math
import tensorflow as tf
from tensorflow import keras

def getitem(lst):
    l = []
    for i in lst:
        for j in i:
            l.append(j)
    return l

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

#the structure of NN
def create_model(x_features):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(15, input_shape=(x_features,)))
    model.add(keras.layers.Dropout(0.5))
    #model.add(keras.layers.Dense(4, input_shape=(4,)))
    #model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1,))
    model.compile(optimizer='adam', loss='mse')
    return model

# win process
df = pd.read_csv('Preprocessing/sheet2.csv', sep=',')
origin_num = df.shape[0]
df1 = df[['age', 'tname', 'jname', 'rating', 'track', 'bardraw', 'besttime', 'going', 'win_t5', 'place_t5']]
train_x = df1.to_numpy()
train_y = np.array(df['ind_win']).astype(np.int)
x_features = train_x.shape[1]
model = create_model(x_features)
history = model.fit(train_x, train_y, epochs=1, verbose=1, validation_split=0.1)

# pla process
train_y_pla = np.array(df['ind_pla']).astype(np.int)
model_pla = create_model(x_features)
history_pla = model_pla.fit(train_x, train_y_pla, epochs=1, verbose=1, validation_split=0.1)

#prediction
dftest = pd.read_csv('Preprocessing/sample.csv', sep=',')
df2 = dftest[['age', 'tname', 'jname', 'rating', 'track', 'bardraw', 'besttime', 'going', 'win_t5', 'place_t5']]
test_x = df2.to_numpy()
#pred_win = model.predict(test_x)
#pred_pla = model_pla.predict(test_x)
dftest['winprob'] = getitem(model.predict(test_x))
dftest['plaprob'] = getitem(model_pla.predict(test_x))
#dftest['winstake'] = np.zeros(len(test_x))
#dftest['plastake'] = np.zeros(len(test_x))

# betting
fixratio = 1/10000
mthresh = 4
mthresh_pla = 1.5
dftest['winstake'] = fixratio * (dftest['winprob'] * dftest['win_t5'] > mthresh)
dftest['plastake'] = fixratio * (dftest['plaprob'] * dftest['place_t5'] > mthresh_pla)

dftest.to_csv('test.csv', index=False, encoding='utf-8')

a1 = dftest['winprob'].to_numpy()
a2 = dftest['ind_win'].to_numpy()
print('RMSEwin: ', rmse(a1, a2))

a3 = dftest['plaprob'].to_numpy()
a4 = dftest['ind_pla'].to_numpy()
print('RMSEpla: ', rmse(a3, a4))

#linear regression
# linear_train={"pickup_latitude":df['pickup_latitude'], "pickup_longitude":df['pickup_longitude']}
# linear_train=pd.DataFrame(linear_train)
# linear_train=linear_train.values.astype(np.int)
# linear_train_y=list(df['fare'])
# X_train, X_test, Y_train, Y_test = train_test_split(linear_train, linear_train_y, train_size=.80)
#
# model_linear = LinearRegression()
# model_linear.fit(X_train, Y_train)
# a = model_linear.intercept_
# b = model_linear.coef_
# print("最佳拟合线:截距", a, ",回归系数：", b)
# score = model_linear.score(X_test, Y_test)


