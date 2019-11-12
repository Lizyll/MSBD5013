import pandas as pd
import geohash
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import librosa.display
import seaborn as sns
import math
import tensorflow as tf
from tensorflow import keras

#the structure of NN
def create_model(x_features):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(4, input_shape=(x_features,)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4, input_shape=(4,)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1,))
    model.compile(optimizer='adam', loss='mse')
    return model

#preprocess
df = pd.read_csv('./tostudent/data4.csv', sep=',')
origin_num = df.shape[0]
df = df[(df['trip_start_timestamp'] != df['trip_end_timestamp']) &
        (df['trip_miles'] != 0.0) &
        (~np.isnan(df['pickup_latitude'])) &
        (~np.isnan(df['pickup_longitude'])) &
        (~np.isnan(df['dropoff_latitude'])) &
        (~np.isnan(df['dropoff_longitude']))
        ]
df['trip_start_timestamp'] = pd.to_datetime(df['trip_start_timestamp'])
df['trip_end_timestamp'] = pd.to_datetime(df['trip_end_timestamp'])

#training
dict_linear_train={"pickup_latitude":df['pickup_latitude'], "pickup_longitude":df['pickup_longitude']}
train_x=pd.DataFrame(dict_linear_train)
train_x=train_x.values.astype(np.float)
train_y = np.array(df['fare']).astype(np.int)
x_features = train_x.shape[1]
model = create_model(x_features)
history = model.fit(train_x, train_y, epochs=1, verbose=1, validation_split=0.1)

#prediction
test_x=train_x
pred_y = model.predict(test_x)
print(pred_y)


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


